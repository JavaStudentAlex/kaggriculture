"""Train KAD-HP-1 (policy.py) on one or two GPUs; the run logic of train.py with this model's inputs,
heads, loss and scores. Usage and arguments as train.py (torchrun --standalone --nproc_per_node=2
policy_train.py --data ... --out ...), which train_job.py runs with --train-script policy_train.py.

Differences from train.py:
- a sample is one turn (policy.PolicyWindows), not a 64-turn context with a 24-turn chunk;
- --init-from is optional: without it the model starts from scratch;
- with --condition the model also reads the team of the seat, one id per team with at least
  --team-min-files seat files in training (the rest share the unknown id 0), hidden like the
  conditions with --team-dropout; the checkpoint keeps the vocabulary and a prompt team (the newest
  day's best-rated team the vocabulary knows) with the conditions' prompt;
- validation reports each head's cross-entropy and accuracy (job, amount, target, now, order,
  order_amount; target, now and amounts given the true job, target and order) and the accuracy of
  whole commands when every head plays its most likely choice (first_active_command_accuracy: the
  commands the seat gave this turn, farmer, hands and market orders, all fields right, as
  KAD-MD-24's metric of the same name; farmer_, hands_, market_ per group). best.pt follows
  val_loss_weighted (val_loss without seat weights), the sum of the heads' cross-entropies.

--tpu trains on every TPU core of the machine (torch_xla; one process per core, started here, not
by torchrun): bfloat16 autocast instead of float16 with a gradient scaler, gradients averaged by
xm.reduce_gradients, batches fed by MpDeviceLoader, and the loop reads values back from the device
(the NaN check, the stop decision, the logged losses) only every --sync-every steps, since each read
waits for the device. Checkpoints hold CPU tensors either way, so a GPU run's last.pt resumes on a
TPU and back (--batch-size is per core: 8 cores x 16 = the 2 GPUs x 64 of the first run).
"""
import argparse
import datetime
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from model import CONDITIONS
from policy import (MARKET_SLOTS, UNITS, JOBS, NOW, ORDERS, Policy, PolicyConfig, PolicyWindows, commands,
                    policy_loss)
from train import (ShuffledWindows, atomic_save, condition_prompt, condition_table, hide_conditions, move, record_day,
                   seat_weights)

PARTS = ('job', 'amount', 'target', 'now', 'order', 'order_amount')
GROUPS = {'farmer': slice(0, 1), 'hands': slice(1, UNITS), 'market': slice(UNITS, UNITS + MARKET_SLOTS)}


def team_vocabulary(dataset, min_files):
    """The teams with at least min_files seat files, most files first (their ids are 1, 2, ...)."""
    counts = Counter(f.get('team') for f in dataset.files if f.get('team'))
    return [team for team, n in counts.most_common() if n >= min_files]


def team_table(dataset, vocabulary):
    """int64 (files,): each file's team id, 0 for a team outside the vocabulary."""
    index = {team: i + 1 for i, team in enumerate(vocabulary)}
    return np.array([index.get(f.get('team'), 0) for f in dataset.files], dtype=np.int64)


def prompt_team(datasets, vocabulary):
    """The best-rated team of the newest day that the vocabulary knows, or None."""
    files = [f for dataset in datasets for f in dataset.files]
    days = [d for d in map(record_day, files) if d]
    if not days:
        return None
    newest, known = max(days), set(vocabulary)
    rated = [(f['team_rating'], f['team']) for f in files if record_day(f) == newest and f.get('team') in known
             and f.get('team_rating') is not None]
    return max(rated)[1] if rated else None


def _item_ce(logits, labels):
    return F.cross_entropy(logits.float().flatten(0, -2), labels.flatten(), reduction='none').view_as(labels)


def on_cpu(obj):
    """obj with every tensor moved to the CPU (a checkpoint that loads on any machine)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: on_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(on_cpu(v) for v in obj)
    return obj


def sync(device):
    """Run the operations queued on an XLA device (a no-op elsewhere)."""
    if device.type == 'xla':
        import torch_xla.core.xla_model as xm
        xm.mark_step()


class Collectives:
    """Sums, minima, maxima and barriers across the training processes: torch.distributed on GPUs,
    torch_xla's own on a TPU (whose cores may be threads of one process, as on a v3-8: 4 processes x
    2 cores), nothing for a single process."""

    def __init__(self, world, xm=None):
        self.world, self.xm = world, xm

    def reduce(self, tensor, op='sum'):
        """The reduction of `tensor` over the processes (a new tensor on a TPU; in place otherwise)."""
        if self.world == 1:
            return tensor
        if self.xm is not None:
            return self.xm.all_reduce({'sum': self.xm.REDUCE_SUM, 'min': self.xm.REDUCE_MIN,
                                       'max': self.xm.REDUCE_MAX}[op], tensor)
        dist.all_reduce(tensor, op={'sum': dist.ReduceOp.SUM, 'min': dist.ReduceOp.MIN, 'max': dist.ReduceOp.MAX}[op])
        return tensor

    def barrier(self, tag='barrier'):
        if self.world > 1:
            self.xm.rendezvous(tag) if self.xm is not None else dist.barrier()


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0, weights=None, conditions=None, teams=None, collectives=None):
    """Mean cross-entropy and accuracy of each head and whole-command accuracy (see the module doc);
    `weights` (one per validation file) adds *_weighted versions in which items count by their
    seat's weight."""
    model.eval()
    total = torch.float32 if device.type == 'xla' else torch.float64   # TPUs have no float64
    table = None if weights is None else torch.as_tensor(weights, dtype=total, device=device)
    # per part: ce sum, right sum, count, weighted ce sum, weighted right sum, weighted count
    stats = torch.zeros(len(PARTS), 6, dtype=total, device=device)
    # per group and overall: right, active, weighted right, weighted active
    whole = torch.zeros(len(GROUPS) + 1, 4, dtype=total, device=device)
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        sync(device)
        b = move(batch, device)
        condition = None if conditions is None else conditions[b['file'].long()]
        team = None if teams is None else teams[b['file'].long()]
        outputs = model(b, condition, team)
        units, market = b['unit_labels'].long(), b['market_labels'].long()
        job, amount, target, now, present = units.unbind(-1)
        present = present.bool()
        order, order_amount = market.unbind(-1)
        labels = (job, amount, target, now, order, order_amount)
        keeps = (present, present & model.job_amount[job], present & (job > 0), present,
                 torch.ones_like(order, dtype=torch.bool), model.order_amount_used[order])
        w = table[b['file'].long()] if table is not None else torch.ones(len(job), dtype=total, device=device)
        for k, (logits, label, keep) in enumerate(zip(outputs, labels, keeps)):
            keep = keep.to(total)
            ce = _item_ce(logits, label).to(total)
            right = (logits.argmax(-1) == label).to(total)
            ww = w.view(-1, *([1] * (keep.dim() - 1))) * keep
            stats[k] += torch.stack([(ce * keep).sum(), (right * keep).sum(), keep.sum(),
                                     (ce * ww).sum(), (right * ww).sum(), ww.sum()])
        job_p, amount_p, _, now_p, orders_p, amounts_p = model.act(b, condition, team)
        chosen = commands(job_p, amount_p, now_p, orders_p, amounts_p)
        truth = b['command'].long()
        active = truth[..., 0] != 0
        correct = (chosen == truth).all(-1) & active
        for g, part in enumerate(list(GROUPS.values()) + [slice(0, UNITS + MARKET_SLOTS)]):
            a, c = active[:, part].to(total), correct[:, part].to(total)
            whole[g] += torch.stack([c.sum(), a.sum(), (c.sum(-1) * w).sum(), (a.sum(-1) * w).sum()])
    sync(device)
    if collectives is not None:
        stats, whole = collectives.reduce(stats), collectives.reduce(whole)
    s, x = stats.cpu().tolist(), whole.cpu().tolist()
    model.train()
    metrics = {}
    for k, name in enumerate(PARTS):
        metrics[f'{name}_ce'] = s[k][0] / max(s[k][2], 1)
        metrics[f'{name}_accuracy'] = s[k][1] / max(s[k][2], 1)
    metrics['val_loss'] = sum(metrics[f'{name}_ce'] for name in PARTS)
    for g, name in enumerate(GROUPS):
        metrics[f'{name}_command_accuracy'] = x[g][0] / max(x[g][1], 1)
    metrics['first_active_command_accuracy'] = x[-1][0] / max(x[-1][1], 1)
    metrics['val_items'] = {name: int(s[k][2]) for k, name in enumerate(PARTS)}
    if table is not None:
        for k, name in enumerate(PARTS):
            metrics[f'{name}_ce_weighted'] = s[k][3] / max(s[k][5], 1e-9)
            metrics[f'{name}_accuracy_weighted'] = s[k][4] / max(s[k][5], 1e-9)
        metrics['val_loss_weighted'] = sum(metrics[f'{name}_ce_weighted'] for name in PARTS)
        metrics['first_active_command_accuracy_weighted'] = x[-1][2] / max(x[-1][3], 1e-9)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=8, help='per rank')
    p.add_argument('--lr', type=float, default=3e-4, help='peak learning rate (cosine to 5%% of it)')
    p.add_argument('--warmup', type=int, default=1000, help='linear warmup steps')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--buffer-files', type=int, default=32, help='files each shuffle worker holds')
    p.add_argument('--stride', type=int, default=7, help='training turn stride, coprime with 24')
    p.add_argument('--val-stride', type=int, default=23, help='validation turn stride, coprime with 24')
    p.add_argument('--val-files', type=int, default=0, help='evenly spaced subset of validation files (0: all)')
    p.add_argument('--val-batches', type=int, default=0, help='cap on validation batches per rank (0: all)')
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--eval-on-start', action='store_true', help='score the starting weights at step 0')
    p.add_argument('--save-every-min', type=float, default=10.0, help='write last.pt at least this often')
    p.add_argument('--max-hours', type=float, default=3.5, help='wall-clock limit of this invocation')
    p.add_argument('--schedule-hours', type=float, default=0.0,
                   help='run the cosine schedule over this much training time, summed across resumes')
    p.add_argument('--rating-halving', type=float, default=0.0, help='as train.py')
    p.add_argument('--weight-floor', type=float, default=0.125, help='as train.py')
    p.add_argument('--margin-doubling', type=float, default=0.0, help='as train.py')
    p.add_argument('--recency-halving', type=float, default=0.0, help='as train.py')
    p.add_argument('--condition', action='store_true', help="read the game's conditions and the seat's team")
    p.add_argument('--condition-dropout', type=float, default=0.15)
    p.add_argument('--condition-drop-all', type=float, default=0.15)
    p.add_argument('--team-min-files', type=int, default=100, help='seat files a team needs for its own id')
    p.add_argument('--team-dropout', type=float, default=0.3, help='chance that a training turn hides the team')
    p.add_argument('--width', type=int, default=384)
    p.add_argument('--layers', type=int, default=8)
    p.add_argument('--heads', type=int, default=6)
    p.add_argument('--history', type=int, default=8, help='earlier turns whose scalars the model reads')
    p.add_argument('--resume')
    p.add_argument('--init-from', nargs='+',
                   help='start from these weights (the best on validation) instead of from scratch')
    p.add_argument('--keep-conditions', action='store_true',
                   help="with --init-from and --condition: keep the first checkpoint's team vocabulary, prompt and "
                        "prompt team instead of deriving them from this run's days (a fine-tune on other data)")
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--tpu', action='store_true', help='train on every TPU core (torch_xla), see the module doc')
    p.add_argument('--sync-every', type=int, default=0,
                   help='steps between reads from the device (NaN check, stop decision, logging); 0: 1, or 20 on TPU')
    args = p.parse_args()
    if args.tpu and os.environ.get('TPU_ACCELERATOR_TYPE', '').endswith('-1'):
        # a one-chip VM (Colab's v5e-1, v6e-1) describes its host as four chips, and xmp.spawn then fails
        # ("Expected 4 worker addresses"): the one core trains in this process
        run(0, args)
    elif args.tpu:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(run, args=(args,))
    else:
        run(None, args)


def run(index, args):
    """One training process: a torchrun rank (GPU), the only process (CPU), or a TPU core (--tpu)."""
    xm = None
    if args.tpu:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        world, rank, local = xr.world_size(), xr.global_ordinal(), xr.local_ordinal()
        device = xm.xla_device()
        hardware = f'{xm.xla_device_hw(device)} (torch_xla {torch_xla.__version__})'
        # torch_xla falls back to the CPU without a TPU, which would spend the schedule's hours on almost no steps
        assert args.smoke or xm.xla_device_hw(device) == 'TPU', f'--tpu, but the device is {hardware}'
    else:
        world = int(os.environ.get('WORLD_SIZE', '1'))
        rank = int(os.environ.get('RANK', '0'))
        local = int(os.environ.get('LOCAL_RANK', '0'))
        if not args.smoke:
            assert torch.cuda.is_available() and world <= torch.cuda.device_count(), \
                'Production run requires one CUDA device per torchrun rank'
        device = torch.device(f'cuda:{local}' if torch.cuda.is_available() and not args.smoke else 'cpu')
        if device.type == 'cuda':
            torch.cuda.set_device(local)
        if world > 1:
            dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
        hardware = torch.cuda.get_device_name(local) if device.type == 'cuda' else None
    collectives = Collectives(world, xm)
    torch.set_num_threads(2)
    sync_every = args.sync_every or (20 if args.tpu else 1)
    random.seed(2026 + rank)
    np.random.seed(2026 + rank)
    torch.manual_seed(2026 + rank)
    if xm is not None:
        xm.set_rng_state(2026 + rank, device)   # dropout on the TPU draws from the device's generator
    print(json.dumps(dict(event='rank_ready', rank=rank, local_rank=local, world_size=world, device=str(device),
                          gpu=hardware)), flush=True)
    train = PolicyWindows(Path(args.data), 'train', stride=args.stride, history=args.history)
    val = PolicyWindows(Path(args.data), 'val', stride=args.val_stride, history=args.history)
    if args.val_files and len(val.files) > args.val_files:
        every = len(val.files) / args.val_files
        val.select([val.files[int(i * every)] for i in range(args.val_files)])
    assert len(train) and len(val), 'Both episode-level train and validation splits required'
    train_weights = val_weights = None
    weighted = bool(args.rating_halving or args.margin_doubling or args.recency_halving)
    if weighted:
        train_weights = seat_weights(train, args.rating_halving, args.weight_floor, args.margin_doubling,
                                     recency_halving=args.recency_halving)
        mean = sum(train_weights) / len(train_weights)
        train_weights = [w / mean for w in train_weights]
        val_weights = seat_weights(val, args.rating_halving, args.weight_floor, args.margin_doubling,
                                   recency_halving=args.recency_halving)
    select = 'val_loss_weighted' if weighted else 'val_loss'
    train_ds = ShuffledWindows(train, rank, world, buffer=args.buffer_files, weights=train_weights)
    loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=device.type == 'cuda', drop_last=True, persistent_workers=args.workers > 0)
    vl = DataLoader(val, batch_size=args.batch_size, sampler=list(range(rank, len(val), world)),
                    num_workers=min(2, args.workers))
    assert len(loader), 'Dataset too small for requested batch size'
    train_conditions = val_conditions = train_teams = val_teams = conditions = None
    vocabulary = []
    if args.condition:
        train_conditions = torch.as_tensor(condition_table(train), device=device)
        val_conditions = torch.as_tensor(condition_table(val), device=device)
        vocabulary = team_vocabulary(train, args.team_min_files)
        started = {}   # the starting model's conditions: its team ids stay, whatever days were added since
        if args.resume or (args.keep_conditions and args.init_from):
            started = torch.load(args.resume or args.init_from[0], map_location='cpu',
                                 weights_only=False).get('conditions') or {}
            vocabulary = list(started.get('teams') or vocabulary)
        train_teams = torch.as_tensor(team_table(train, vocabulary), device=device)
        val_teams = torch.as_tensor(team_table(val, vocabulary), device=device)
        conditions = condition_prompt(train, val)
        conditions.update(teams=vocabulary, prompt_team=prompt_team((train, val), vocabulary))
        if args.keep_conditions and started.get('prompt'):
            conditions.update(prompt=started['prompt'], prompt_team=started.get('prompt_team'))
    config = PolicyConfig(width=args.width, layers=args.layers, heads=args.heads, history=args.history,
                          condition_dim=len(CONDITIONS) if args.condition else 0, teams=len(vocabulary))
    if args.smoke:
        config.width, config.layers, config.heads, config.market_layers = 48, 2, 4, 1
    model = Policy(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')   # off on a TPU (bfloat16)
    autocast = dict(device_type=device.type, dtype=torch.bfloat16 if device.type == 'xla' else torch.float16,
                    enabled=device.type in ('cuda', 'xla'))
    val_batches = 2 if args.smoke else args.val_batches

    def validate():
        return evaluate(model, vl, device, val_batches, val_weights, val_conditions, val_teams, collectives)

    start, best, trained_s, initial = 0, float('inf'), 0.0, None
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        assert asdict(PolicyConfig(**state['config'])) == model.config_dict(), 'the --resume checkpoint has another config'
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        if state.get('scaler'):   # empty from a TPU run (no scaler there)
            scaler.load_state_dict(state['scaler'])
        start, best, trained_s = state['step'], state['best_val'], state.get('trained_s', 0.0)
    elif args.init_from:
        candidates = []
        for path in args.init_from:
            state = torch.load(path, map_location='cpu', weights_only=False)
            if state.get('config') != model.config_dict():
                if rank == 0:
                    print(json.dumps(dict(event='init_skipped', path=path, reason='another model config')), flush=True)
                continue
            candidates.append(dict(path=path, step=state.get('step'), recorded=state.get('metrics'), model=state['model']))
        for candidate in candidates if len(candidates) > 1 else []:
            model.load_state_dict(candidate['model'])
            candidate['metrics'] = validate()
            if rank == 0:
                print(json.dumps(dict(event='init_candidate', path=candidate['path'], step=candidate['step'],
                                      metrics=candidate['metrics'])), flush=True)
        if candidates:
            initial = min(candidates, key=lambda c: c['metrics'][select]) if len(candidates) > 1 else candidates[0]
            model.load_state_dict(initial['model'])
            if rank == 0:
                print(json.dumps(dict(event='init_from', path=initial['path'], step=initial['step'],
                                      metrics=initial['recorded'])), flush=True)
    # on a TPU the gradients are averaged by xm.reduce_gradients instead of DDP
    net = DDP(model, device_ids=[local] if device.type == 'cuda' else None) if world > 1 and xm is None else model
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy = dict(jobs=[list(j) for j in JOBS], orders=[list(o) for o in ORDERS], now=list(NOW))
    if rank == 0:
        metadata = dict(model_name='KAD-HP-1', policy='structured multi-head policy (policy.py)',
                        pretrained_checkpoint=initial and initial['path'], config=model.config_dict(),
                        parameters=sum(x.numel() for x in model.parameters()), world_size=world,
                        train_turns=len(train), validation_turns=len(val), train_files=len(train.files),
                        validation_files=len(val.files), args=vars(args), best_checkpoint_metric=select)
        if conditions:
            metadata['conditions'] = dict(conditions, teams=len(vocabulary))
        if not args.resume or not (out / 'config.json').exists():
            (out / 'config.json').write_text(json.dumps(metadata, indent=2))
        print(json.dumps(dict(event='training_start', start_step=start, **metadata)), flush=True)

    def spent():
        return trained_s + (time.monotonic() - elapsed_start)

    def save_last(step, metrics):
        atomic_save(dict(config=model.config_dict(), model=on_cpu(model.state_dict()), optimizer=on_cpu(opt.state_dict()),
                         scaler=scaler.state_dict(), step=step, best_val=best, metrics=metrics, trained_s=spent(),
                         conditions=conditions, policy=policy), out / 'last.pt')

    def log(row):
        print(json.dumps(row), flush=True)
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')

    metrics = {}
    if args.eval_on_start and not args.resume:
        metrics = (initial or {}).get('metrics') or validate()
        if rank == 0:
            log(dict(event='validation', step=0, **metrics))
    elapsed_start = time.monotonic()
    last_save = last_log = time.monotonic()
    logged_step = start
    epoch = start // len(loader)
    train_ds.set_epoch(epoch)
    if xm is not None:
        import torch_xla.distributed.parallel_loader as pl
        feed = pl.MpDeviceLoader(loader, device)   # copies the next batches to the core while it computes
    else:
        feed = loader
    iterator = iter(feed)
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
            train_ds.set_epoch(epoch)
            iterator = iter(feed)
            batch = next(iterator)
        b = move(batch, device)
        condition = team = None
        if train_conditions is not None:
            files = b['file'].long()
            condition = hide_conditions(train_conditions[files], args.condition_dropout, args.condition_drop_all)
            team = train_teams[files] * (torch.rand(len(files), device=device) >= args.team_dropout)
        done = progress(step)
        lr_factor = min(1., step / args.warmup) * .5 * (1 + math.cos(math.pi * min(done, 1.)))
        for group in opt.param_groups:
            group['lr'] = args.lr * max(lr_factor, .05)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(**autocast):
            loss, parts = policy_loss(model, net(b, condition, team), b)
        scaler.scale(loss).backward()
        if xm is not None and world > 1:
            xm.reduce_gradients(opt)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        scaler.step(opt)
        scaler.update()
        sync(device)
        # reading from the device waits for it: every rank checks on the same steps, every sync_every
        # (then this step's loss stands for the ones before it: a NaN stays once in the weights)
        check = step % sync_every == 0 or step % args.eval_every == 0 or step == start + 1
        stop = torch.zeros((), dtype=torch.bool)
        if check:
            finite = collectives.reduce(torch.isfinite(loss.detach()).int(), 'min')
            if not finite.item():
                raise FloatingPointError(f'Nonfinite training loss at {step}')
            stop = collectives.reduce(torch.tensor([int(time.monotonic() - elapsed_start >= args.max_hours * 3600),
                                                    int(done >= 1.)], device=device), 'max')
            finished = bool(stop[1].item())
            stop = stop.max().cpu()
        if rank == 0 and check and (step == start + 1 or step % 20 == 0):
            now = time.monotonic()
            log(dict(event='train', step=step, epoch=epoch, loss=float(loss.detach()),
                     **{k: round(float(v), 4) for k, v in parts.items()}, lr=opt.param_groups[0]['lr'],
                     elapsed_s=round(now - elapsed_start, 2),
                     turns_per_s=round((step - logged_step) * args.batch_size * world / max(now - last_log, 1e-9), 1),
                     peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0))
            last_log, logged_step = now, step
        if step % args.eval_every == 0 or stop.item():
            metrics = validate()
            if rank == 0:
                log(dict(event='validation', step=step, **metrics))
                improved = metrics[select] < best
                best = min(best, metrics[select])
                save_last(step, metrics)
                last_save = time.monotonic()
                if improved:
                    atomic_save(dict(config=model.config_dict(), model=on_cpu(model.state_dict()), step=step,
                                     metrics=metrics, conditions=conditions, policy=policy), out / 'best.pt')
                (out / 'scores.json').write_text(json.dumps(dict(step=step, trained_s=round(spent()), complete=finished,
                                                                 **metrics), indent=2))
            collectives.barrier('validation')
        elif rank == 0 and time.monotonic() - last_save >= args.save_every_min * 60:
            save_last(step, metrics)
            last_save = time.monotonic()
        if stop.item():
            break
    collectives.barrier('end')
    if world > 1 and xm is None:
        dist.destroy_process_group()
    if rank == 0:
        if finished:
            (out / 'complete').write_text(json.dumps(dict(step=step, trained_s=round(spent()))) + '\n')
        print(json.dumps(dict(event='TRAINING_COMPLETE' if finished else 'TRAINING_STOPPED', step=step,
                              trained_s=round(spent()), **metrics)), flush=True)


if __name__ == '__main__':
    main()
