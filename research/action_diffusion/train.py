"""Bounded, resumable two-device DDP training. Never touches the existing TTM.
Usage: torchrun --standalone --nproc_per_node=2 train.py --data ... --out ...
CPU --smoke mode is solely a functional test, not a strength benchmark.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from model import ActionDiffusion, Config, corrupt, masked_loss, predict


def move(batch, device):
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


def atomic_save(state, path):
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=24):
    model.eval()
    # fixed fully masked readout: comparable across checkpoints, no random masks
    totals = torch.zeros(8, dtype=torch.float64, device=device)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move(batch, device)
        y = batch['actions'].long()
        noisy = model.sizes[None, None, :].expand_as(y)
        logits = model(batch['context'].float(), batch['context_mask'].bool(), noisy,
                       torch.ones(len(y), device=device))
        valid = batch['action_mask'].bool()[:, :, None].expand_as(y)
        loss = masked_loss(logits, y, torch.ones_like(y, dtype=torch.bool), batch['action_mask'].bool())
        pred = logits.argmax(-1)
        active = (y[:, :, 0::3] != 0) & batch['action_mask'][:, :, None].bool()
        correct_op = pred[:, :, 0::3] == y[:, :, 0::3]
        command_correct = (pred.reshape(*pred.shape[:2], -1, 3) == y.reshape(*y.shape[:2], -1, 3)).all(-1)
        totals += torch.stack([loss.double(), torch.ones((), device=device),
                               ((pred == y) & valid).sum(), valid.sum(),
                               (correct_op & active).sum(), active.sum(),
                               (command_correct & active).sum(), active.sum()])
    if dist.is_initialized():
        dist.all_reduce(totals)
    v = totals.cpu().tolist()
    model.train()
    return dict(val_masked_ce=v[0]/max(v[1],1), field_accuracy=v[2]/max(v[3],1),
                active_operation_accuracy=v[4]/max(v[5],1), active_command_accuracy=v[6]/max(v[7],1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=8, help='per rank')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--max-hours', type=float, default=3.5)
    p.add_argument('--resume')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    from data import WindowDataset, FEATURE_DIM, field_sizes
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    if not args.smoke:
        assert torch.cuda.is_available() and torch.cuda.device_count() == 2 and world == 2, \
            'Production run requires exactly two CUDA devices and two torchrun ranks'
    device = torch.device(f'cuda:{local}' if torch.cuda.is_available() and not args.smoke else 'cpu')
    torch.set_num_threads(2)
    if device.type == 'cuda':
        torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    random.seed(2026 + rank)
    np.random.seed(2026 + rank)
    torch.manual_seed(2026 + rank)
    print(json.dumps(dict(event='rank_ready', rank=rank, local_rank=local, world_size=world,
                          device=str(device), gpu=torch.cuda.get_device_name(local) if device.type=='cuda' else None)), flush=True)
    train = WindowDataset(Path(args.data), 'train')
    val = WindowDataset(Path(args.data), 'val', stride=24)
    assert len(train) and len(val), 'Both episode-level train and validation splits required'
    ts = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True, seed=2026)
    # Non-padded validation partition, avoiding duplicated evaluation windows.
    vs = list(range(rank, len(val), world))
    loader = DataLoader(train, batch_size=args.batch_size, sampler=ts, num_workers=args.workers,
                        pin_memory=device.type=='cuda', drop_last=True, persistent_workers=args.workers>0)
    vl = DataLoader(val, batch_size=args.batch_size, sampler=vs, num_workers=0)
    assert len(loader), 'Dataset too small for requested batch size'
    config = Config(FEATURE_DIM, list(field_sizes))
    if args.smoke:
        config.width, config.layers, config.heads = 48, 2, 4
    model = ActionDiffusion(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
    start = 0
    best = float('inf')
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        assert state['config'] == model.config_dict()
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start = state['step']
        best = state['best_val']
    net = DDP(model, device_ids=[local] if device.type=='cuda' else None) if world>1 else model
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        metadata = dict(model_name='KAD-MD-24', diffusion='conditional absorbing-mask discrete diffusion',
                        pretrained_checkpoint=None, config=model.config_dict(),
                        parameters=sum(x.numel() for x in model.parameters()), world_size=world,
                        train_windows=len(train), validation_windows=len(val), args=vars(args))
        (out/'config.json').write_text(json.dumps(metadata, indent=2))
        print(json.dumps(dict(event='training_start', **metadata)), flush=True)
    elapsed_start = time.monotonic()
    epoch = start//len(loader)
    ts.set_epoch(epoch)
    iterator = iter(loader)
    net.train()
    metrics = {}
    for step in range(start+1, args.steps+1):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            ts.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        b = move(batch, device)
        y = b['actions'].long()
        noisy, mask, fraction = corrupt(y, model.sizes)
        lr_factor = min(1., step/100) * .5*(1+math.cos(math.pi*min(step,args.steps)/args.steps))
        for group in opt.param_groups:
            group['lr'] = 2e-4*max(lr_factor, .05)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
            logits = net(b['context'].float(), b['context_mask'].bool(), noisy, fraction)
            loss = masked_loss(logits, y, mask, b['action_mask'].bool())
        finite = torch.tensor(int(torch.isfinite(loss)), device=device)
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError(f'Nonfinite training loss at {step}')
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        scaler.step(opt)
        scaler.update()
        stop = torch.tensor(int(time.monotonic()-elapsed_start >= args.max_hours*3600), device=device)
        if world>1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if rank==0 and (step == 1 or step % 20 == 0):
            row = dict(event='train', step=step, epoch=epoch, loss=float(loss.detach()),
                       elapsed_s=round(time.monotonic()-elapsed_start,2),
                       peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0)
            print(json.dumps(row), flush=True)
            with (out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
        if step % args.eval_every == 0 or step==args.steps or stop.item():
            metrics = evaluate(model, vl, device, max_batches=2 if args.smoke else 24)
            if rank==0:
                print(json.dumps(dict(event='validation', step=step, **metrics)), flush=True)
                improved = metrics['val_masked_ce'] < best
                best = min(best, metrics['val_masked_ce'])
                state = dict(config=model.config_dict(), model=model.state_dict(), optimizer=opt.state_dict(),
                             scaler=scaler.state_dict(), step=step, best_val=best, metrics=metrics)
                atomic_save(state, out/'last.pt')
                if improved:
                    atomic_save(dict(config=model.config_dict(), model=model.state_dict(), step=step,
                                     metrics=metrics), out/'best.pt')
                (out/'scores.json').write_text(json.dumps(dict(step=step, **metrics), indent=2))
            if world>1:
                dist.barrier()
        if stop.item():
            break
    if world>1:
        dist.barrier()
        dist.destroy_process_group()
    if rank==0:
        print(json.dumps(dict(event='TRAINING_COMPLETE', step=step, **metrics)), flush=True)


if __name__ == '__main__':
    main()
