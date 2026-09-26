"""Bounded, resumable training on one or two GPUs. Never touches the existing TTM.
Usage: torchrun --standalone --nproc_per_node=<gpus> train.py --data ... --out ...
       (one GPU: python train.py --data ... --out ...)
CPU --smoke mode is solely a functional test, not a strength benchmark.

--init-from <checkpoint> starts from trained weights (e.g. runs/action_diffusion_run/best.pt)
with a fresh optimizer and learning-rate schedule; --resume <last.pt> continues a run exactly.
--eval-on-start scores the starting weights first (step 0), the number training must beat.
Validation reports the executed action separately: the first_* metrics score horizon slot 0,
and first_command_accuracy_by_hour splits it by game hour (anchor step % 24).
Seat weights (seat_weights) favour good play: --rating-halving N samples a seat in proportion
to 2 ** ((its team's rating - the day's best team rating) / N), and --margin-doubling M times
2 ** ((its cash lead over the opponent - the lead its rating edge predicts) / M), leads as a
fraction of the game's mean cash. With either, validation also reports *_weighted metrics and
best.pt follows val_masked_ce_weighted.
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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset, get_worker_info

from model import ActionDiffusion, Config, corrupt, masked_loss, predict


def move(batch, device):
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


def atomic_save(state, path):
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def seat_weights(dataset, halving=0.0, floor=0.125, margin_doubling=0.0, cap=4.0):
    """Relative sampling weight of each file (one seat of one game) of a WindowDataset: how good
    the seat's team was that day times how well the seat played that game (data.add_ratings).

    Team (halving > 0): 2 ** ((team_rating - best) / halving), where best is the top team rating
    of the file's day (its source): the day's best team weighs 1, a team `halving` points below
    it 1/2, none less than `floor`; unrated files take the mean of the rated ones.
    Game (margin_doubling > 0): 2 ** ((margin - expected_margin) / margin_doubling), between
    1/cap and cap, where margin is the seat's cash lead over its opponent as a fraction of their
    mean cash and expected_margin the lead its rating edge predicts. Beating that by
    margin_doubling (0.05: 5% of the game's cash) doubles the weight; 1 when unknown. Absolute
    cash is not used: the game, not the player, sets it (both seats end with similar cash).
    """
    best = {}
    for record in dataset.manifest['files']:
        if record.get('team_rating') is not None:
            best[record['source']] = max(best.get(record['source'], -math.inf), record['team_rating'])
    team = [max(floor, 2 ** ((f['team_rating'] - best[f['source']]) / halving))
            if halving and f.get('team_rating') is not None else None for f in dataset.files]
    rated = [w for w in team if w is not None]
    fill = sum(rated) / len(rated) if rated else 1.0
    weights = []
    for f, w in zip(dataset.files, team):
        game = 1.0
        if margin_doubling and f.get('margin') is not None and f.get('expected_margin') is not None:
            game = min(cap, max(1 / cap, 2 ** ((f['margin'] - f['expected_margin']) / margin_doubling)))
        weights.append((fill if w is None else w) * game)
    return weights


class ShuffledWindows(IterableDataset):
    """Training windows with each file decompressed once per epoch.

    WindowDataset's random access decompresses a whole seat file (17 MB, 35-47 ms) for every
    window, which starves a GPU. Here each DataLoader worker takes its share of the files
    (split over ranks and workers), holds `buffer` of them in memory and draws windows
    uniformly from the held files, reading the next file when one is used up. Every window
    comes once per epoch; a batch mixes windows of up to `buffer` files.

    With `weights` (one per file, mean 1) a file contributes weight x its windows per epoch:
    each window int(weight) times, plus once more with probability equal to the fraction.
    """

    def __init__(self, windows, rank, world, seed=2026, buffer=32, weights=None):
        self.windows, self.rank, self.world, self.seed, self.buffer = windows, rank, world, seed, buffer
        self.weights = weights
        self.epoch = 0
        self.passes = 0   # persistent workers keep their copy, so they count epochs themselves

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.windows) // self.world

    def __iter__(self):
        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        epoch = self.epoch + self.passes
        self.passes += 1
        order = np.random.default_rng([self.seed, epoch]).permutation(len(self.windows.files))
        mine = iter(order[self.rank * workers + worker::self.world * workers])
        rng = np.random.default_rng([self.seed, epoch, self.rank, worker])
        held = []

        def fill():
            while len(held) < self.buffer:
                index = next(mine, None)
                if index is None:
                    return
                X, A = self.windows.read(int(index))
                anchors = list(self.windows.anchors(len(X)))
                if self.weights is not None:
                    weight = self.weights[int(index)]
                    extra = rng.random(len(anchors)) < weight - int(weight)
                    anchors = anchors * int(weight) + [t for t, keep in zip(anchors, extra) if keep]
                    if not anchors:
                        continue
                rng.shuffle(anchors)
                held.append((X, A, anchors))

        fill()
        while held:
            k = int(rng.integers(len(held)))
            X, A, anchors = held[k]
            yield self.windows.window(X, A, anchors.pop())
            if not anchors:
                held[k] = held[-1]
                held.pop()
                fill()


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0, weights=None):
    """Fixed fully masked readout: comparable across checkpoints, no random masks.

    `weights` (one per validation file, see seat_weights) adds *_weighted metrics in which each
    window counts by its seat's weight; the unweighted metrics are unchanged.
    """
    model.eval()
    totals = torch.zeros(14, dtype=torch.float64, device=device)
    weighted = torch.zeros(6, dtype=torch.float64, device=device)
    table = None if weights is None else torch.as_tensor(weights, dtype=torch.float64, device=device)
    by_hour = torch.zeros(2, 24, dtype=torch.float64, device=device)
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = move(batch, device)
        y = batch['actions'].long()
        amask = batch['action_mask'].bool()
        noisy = model.sizes[None, None, :].expand_as(y)
        logits = model(batch['context'].float(), batch['context_mask'].bool(), noisy,
                       torch.ones(len(y), device=device))
        valid = amask[:, :, None].expand_as(y)
        loss = masked_loss(logits, y, torch.ones_like(y, dtype=torch.bool), amask)
        pred = logits.argmax(-1)
        active = (y[:, :, 0::3] != 0) & amask[:, :, None]
        correct_op = pred[:, :, 0::3] == y[:, :, 0::3]
        command_correct = (pred.reshape(*pred.shape[:2], -1, 3) == y.reshape(*y.shape[:2], -1, 3)).all(-1)
        # slot 0 is the action the agent executes before it replans
        a0, o0, c0 = active[:, 0], correct_op[:, 0] & active[:, 0], command_correct[:, 0] & active[:, 0]
        exact0 = (pred[:, 0] == y[:, 0]).all(-1) & amask[:, 0]
        totals += torch.stack([loss.double(), torch.ones((), device=device, dtype=torch.float64),
                               ((pred == y) & valid).sum(), valid.sum(),
                               (correct_op & active).sum(), active.sum(),
                               (command_correct & active).sum(), active.sum(),
                               o0.sum(), a0.sum(), c0.sum(), a0.sum(),
                               exact0.sum(), amask[:, 0].sum()]).double()
        hour = batch['anchor'].long() % 24
        by_hour[0].index_add_(0, hour, c0.sum(-1).double())
        by_hour[1].index_add_(0, hour, a0.sum(-1).double())
        if table is not None:
            w = table[batch['file'].long()]
            ce = F.cross_entropy(logits.float().flatten(0, 2), y.flatten(), reduction='none').view_as(y)
            fields = (y[:, :, 0::3] != 0).repeat_interleave(3, -1)   # as masked_loss: every operation,
            fields[:, :, 0::3] = True                                 # the arguments of active commands
            fields &= valid
            weighted += torch.stack([(w * (ce * fields).sum((1, 2))).sum(), (w * fields.sum((1, 2))).sum(),
                                     (w * (command_correct & active).sum((1, 2))).sum(), (w * active.sum((1, 2))).sum(),
                                     (w * c0.sum(-1)).sum(), (w * a0.sum(-1)).sum()])
    if dist.is_initialized():
        dist.all_reduce(totals)
        dist.all_reduce(by_hour)
        dist.all_reduce(weighted)
    v = totals.cpu().tolist()
    h = by_hour.cpu().tolist()
    model.train()
    metrics = dict(val_masked_ce=v[0]/max(v[1], 1), field_accuracy=v[2]/max(v[3], 1),
                   active_operation_accuracy=v[4]/max(v[5], 1), active_command_accuracy=v[6]/max(v[7], 1),
                   first_active_operation_accuracy=v[8]/max(v[9], 1),
                   first_active_command_accuracy=v[10]/max(v[11], 1),
                   first_action_exact=v[12]/max(v[13], 1), val_windows=int(v[13]),
                   first_command_accuracy_by_hour=[round(c/t, 4) if t else None for c, t in zip(*h)])
    if table is not None:
        u = weighted.cpu().tolist()
        metrics.update(val_masked_ce_weighted=u[0]/max(u[1], 1e-9), active_command_accuracy_weighted=u[2]/max(u[3], 1e-9),
                       first_active_command_accuracy_weighted=u[4]/max(u[5], 1e-9))
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=8, help='per rank')
    p.add_argument('--lr', type=float, default=2e-4, help='peak learning rate (cosine to 5%% of it)')
    p.add_argument('--warmup', type=int, default=100, help='linear warmup steps')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--loader', choices=('shuffle', 'random'), default='shuffle',
                   help='shuffle: files read once per epoch (ShuffledWindows); random: the original random access')
    p.add_argument('--buffer-files', type=int, default=32, help='files each shuffle worker holds')
    p.add_argument('--stride', type=int, default=7, help='training anchor stride, coprime with 24')
    p.add_argument('--val-stride', type=int, default=23, help='validation anchor stride, coprime with 24')
    p.add_argument('--val-files', type=int, default=0, help='evenly spaced subset of validation files (0: all)')
    p.add_argument('--val-batches', type=int, default=0, help='cap on validation batches per rank (0: all)')
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--eval-on-start', action='store_true', help='score the starting weights at step 0')
    p.add_argument('--save-every-min', type=float, default=10.0, help='write last.pt at least this often')
    p.add_argument('--max-hours', type=float, default=3.5, help='wall-clock limit of this invocation')
    p.add_argument('--schedule-hours', type=float, default=0.0,
                   help='run the cosine schedule over this much training time, summed across resumes, '
                        'instead of over --steps; training is complete when it is used up (<out>/complete)')
    p.add_argument('--rating-halving', type=float, default=0.0,
                   help="favour the day's top teams: a seat's windows are sampled in proportion to 2 ** ((its "
                        "team's rating - the day's best team rating) / this), so 50 halves them per 50 points "
                        "below the best team. 0: every team alike")
    p.add_argument('--weight-floor', type=float, default=0.125, help="lowest standing weight of a team, relative to the day's best")
    p.add_argument('--margin-doubling', type=float, default=0.0,
                   help="favour seats that played their game well: times 2 ** ((cash lead over the opponent - "
                        "the lead the seat's rating edge predicts) / this), leads as a fraction of the game's mean "
                        "cash, capped at 4x either way; 0.05 doubles a seat 5%% ahead of expectation. With this "
                        "or --rating-halving, validation adds *_weighted metrics and best.pt follows "
                        "val_masked_ce_weighted. 0: off")
    p.add_argument('--resume')
    p.add_argument('--init-from', help='start from these weights with a fresh optimizer and schedule')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    from data import WindowDataset, FEATURE_DIM, field_sizes
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    if not args.smoke:
        assert torch.cuda.is_available() and world <= torch.cuda.device_count(), \
            'Production run requires one CUDA device per torchrun rank'
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
    train = WindowDataset(Path(args.data), 'train', stride=args.stride)
    val = WindowDataset(Path(args.data), 'val', stride=args.val_stride)
    if args.val_files and len(val.files) > args.val_files:
        every = len(val.files) / args.val_files
        val.select([val.files[int(i * every)] for i in range(args.val_files)])
    assert len(train) and len(val), 'Both episode-level train and validation splits required'
    train_weights = val_weights = None
    weighted = bool(args.rating_halving or args.margin_doubling)
    if weighted:
        assert args.loader == 'shuffle', 'seat weights need the shuffle loader'
        train_weights = seat_weights(train, args.rating_halving, args.weight_floor, args.margin_doubling)
        mean = sum(train_weights) / len(train_weights)
        train_weights = [w / mean for w in train_weights]
        val_weights = seat_weights(val, args.rating_halving, args.weight_floor, args.margin_doubling)
    select = 'val_masked_ce_weighted' if weighted else 'val_masked_ce'
    if args.loader == 'shuffle':
        train_ds = ShuffledWindows(train, rank, world, buffer=args.buffer_files, weights=train_weights)
        loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=device.type=='cuda', drop_last=True, persistent_workers=args.workers>0)
        set_epoch = train_ds.set_epoch
    else:
        ts = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True, seed=2026)
        loader = DataLoader(train, batch_size=args.batch_size, sampler=ts, num_workers=args.workers,
                            pin_memory=device.type=='cuda', drop_last=True, persistent_workers=args.workers>0)
        set_epoch = ts.set_epoch
    # Non-padded validation partition, avoiding duplicated evaluation windows; in order, so
    # each file is decompressed once.
    vs = list(range(rank, len(val), world))
    vl = DataLoader(val, batch_size=args.batch_size, sampler=vs, num_workers=min(2, args.workers))
    assert len(loader), 'Dataset too small for requested batch size'
    config = Config(FEATURE_DIM, list(field_sizes))
    if args.smoke:
        config.width, config.layers, config.heads = 48, 2, 4
    model = ActionDiffusion(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
    start = 0
    best = float('inf')
    trained_s = 0.0   # training time of earlier invocations (--schedule-hours)
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        assert state['config'] == model.config_dict()
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start = state['step']
        best = state['best_val']
        trained_s = state.get('trained_s', 0.0)
    elif args.init_from:
        state = torch.load(args.init_from, map_location=device, weights_only=False)
        assert state['config'] == model.config_dict(), 'the --init-from checkpoint has another model config'
        model.load_state_dict(state['model'])
        if rank == 0:
            print(json.dumps(dict(event='init_from', path=args.init_from, step=state.get('step'),
                                  metrics=state.get('metrics'))), flush=True)
    net = DDP(model, device_ids=[local] if device.type=='cuda' else None) if world>1 else model
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        metadata = dict(model_name='KAD-MD-24', diffusion='conditional absorbing-mask discrete diffusion',
                        pretrained_checkpoint=args.init_from, config=model.config_dict(),
                        parameters=sum(x.numel() for x in model.parameters()), world_size=world,
                        train_windows=len(train), validation_windows=len(val),
                        train_files=len(train.files), validation_files=len(val.files), args=vars(args),
                        best_checkpoint_metric=select)
        if train_weights is not None:
            # share of the training windows of the day's top teams, of game winners and of seats that beat
            # their expected margin, before and after weighting
            ranks = [f.get('team_rank') for f in train.files]
            share = {}
            won = [(f.get('margin') or 0) > 0 for f in train.files]
            beat = [f.get('margin') is not None and f.get('expected_margin') is not None
                    and f['margin'] > f['expected_margin'] for f in train.files]
            for name, inside in (('top3', [r is not None and r <= 3 for r in ranks]),
                                 ('top10', [r is not None and r <= 10 for r in ranks]), ('winners', won),
                                 ('above_expectation', beat)):
                share[name] = dict(uniform=round(sum(inside) / len(inside), 3),
                                   weighted=round(sum(w for w, i in zip(train_weights, inside) if i) / len(inside), 3))
            metadata['seat_weights'] = dict(rated_files=sum(r is not None for r in ranks), share=share)
        if not args.resume or not (out/'config.json').exists():
            (out/'config.json').write_text(json.dumps(metadata, indent=2))
        print(json.dumps(dict(event='training_start', start_step=start, **metadata)), flush=True)

    def spent():
        return trained_s + (time.monotonic() - elapsed_start)

    def save_last(step, metrics):
        atomic_save(dict(config=model.config_dict(), model=model.state_dict(), optimizer=opt.state_dict(),
                         scaler=scaler.state_dict(), step=step, best_val=best, metrics=metrics,
                         trained_s=spent()), out/'last.pt')

    metrics = {}
    if args.eval_on_start and not args.resume:
        metrics = evaluate(model, vl, device, max_batches=2 if args.smoke else args.val_batches, weights=val_weights)
        if rank == 0:
            print(json.dumps(dict(event='validation', step=0, **metrics)), flush=True)
            with (out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(dict(event='validation', step=0, **metrics)) + '\n')
    elapsed_start = time.monotonic()
    last_save = last_log = time.monotonic()
    logged_step = start
    epoch = start//len(loader)
    set_epoch(epoch)
    iterator = iter(loader)
    net.train()
    step = start
    budget = args.schedule_hours * 3600

    def progress(step):
        return spent() / budget if budget else step / args.steps

    finished = progress(start) >= 1.
    while not finished:
        step += 1
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        b = move(batch, device)
        y = b['actions'].long()
        noisy, mask, fraction = corrupt(y, model.sizes)
        done = progress(step)
        lr_factor = min(1., step/args.warmup) * .5*(1+math.cos(math.pi*min(done, 1.)))
        for group in opt.param_groups:
            group['lr'] = args.lr*max(lr_factor, .05)
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
        stop = torch.tensor([int(time.monotonic()-elapsed_start >= args.max_hours*3600), int(done >= 1.)],
                            device=device)
        if world>1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        finished = bool(stop[1].item())
        stop = stop.max()
        if rank==0 and (step == 1 or step % 20 == 0):
            now = time.monotonic()
            row = dict(event='train', step=step, epoch=epoch, loss=float(loss.detach()),
                       lr=opt.param_groups[0]['lr'], elapsed_s=round(now-elapsed_start, 2),
                       windows_per_s=round((step-logged_step)*args.batch_size*world/max(now-last_log, 1e-9), 1),
                       peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0)
            last_log, logged_step = now, step
            print(json.dumps(row), flush=True)
            with (out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
        if step % args.eval_every == 0 or stop.item():
            metrics = evaluate(model, vl, device, max_batches=2 if args.smoke else args.val_batches, weights=val_weights)
            if rank==0:
                print(json.dumps(dict(event='validation', step=step, **metrics)), flush=True)
                with (out/'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(dict(event='validation', step=step, **metrics)) + '\n')
                improved = metrics[select] < best
                best = min(best, metrics[select])
                save_last(step, metrics)
                last_save = time.monotonic()
                if improved:
                    atomic_save(dict(config=model.config_dict(), model=model.state_dict(), step=step,
                                     metrics=metrics), out/'best.pt')
                (out/'scores.json').write_text(json.dumps(dict(step=step, trained_s=round(spent()), complete=finished,
                                                               **metrics), indent=2))
            if world>1:
                dist.barrier()
        elif rank == 0 and time.monotonic() - last_save >= args.save_every_min * 60:
            save_last(step, metrics)
            last_save = time.monotonic()
        if stop.item():
            break
    if world>1:
        dist.barrier()
        dist.destroy_process_group()
    if rank==0:
        if finished:
            (out/'complete').write_text(json.dumps(dict(step=step, trained_s=round(spent()))) + '\n')
        print(json.dumps(dict(event='TRAINING_COMPLETE' if finished else 'TRAINING_STOPPED', step=step,
                              trained_s=round(spent()), **metrics)), flush=True)


if __name__ == '__main__':
    main()
