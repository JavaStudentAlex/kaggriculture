"""Export KAD-HP-1 for the graph runtime's numpy copilot (hazel_runtime/kad/kad_numpy.py) and check the port
(2026-09-29; run on cliproxyapi under a python with torch, e.g. ~/kagg-colab/ad_check/venv-torch/bin/python).

    python export_kad.py --code <research/action_diffusion> CHECKPOINT OUT.npz [--replay DAY.zip --games 2 --turns 80]

Writes every float parameter of the Policy as float16, plus `token_bias` (what the prompt adds to every token:
PolicyPlayer's default prompt, i.e. the checkpoint's prompt conditions and prompt team) and `__meta__` (config,
prompt, team, the checkpoint's sha256). With --replay it takes both seats of the first --games games of a published
day zip and, on --turns turns of each spread over the game, compares the numpy model with the torch model: the
largest difference of the order-class and job-class probabilities, and whether the greedy orders and jobs agree
(numpy float32 from the checkpoint, then float16 from OUT.npz). It also checks that KadAdvisor.see builds the same
observation rows as data.episode_arrays, and times KadAdvisor.advise on one thread.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent / 'hazel_runtime'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('out')
    ap.add_argument('--code', required=True, help='research/action_diffusion (policy.py, play.py, model.py, data.py)')
    ap.add_argument('--replay', default=None)
    ap.add_argument('--games', type=int, default=2)
    ap.add_argument('--turns', type=int, default=80)
    args = ap.parse_args()
    sys.path.insert(0, args.code)
    import torch
    torch.set_num_threads(1)
    from play import PolicyPlayer
    import policy as P
    import data as AD

    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    player = PolicyPlayer(saved)
    model = player.model.eval()
    c = model.config
    with torch.no_grad():
        bias = torch.zeros(c.width)
        if c.condition_dim and player.condition is not None:
            bias = bias + model.condition(player.condition.float())[0]
        if c.teams and player.team is not None:
            bias = bias + model.team(player.team.long())[0]
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    conditions = saved.get('conditions') or {}
    meta = {'config': asdict(c), 'prompt': conditions.get('prompt'), 'team': conditions.get('prompt_team'),
            'checkpoint': Path(args.checkpoint).name,
            'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            'orders': len(P.ORDERS), 'jobs': len(P.JOBS), 'step': saved.get('step')}
    np.savez(args.out, **{k: v.astype(np.float16) for k, v in state.items()},
             token_bias=bias.numpy().astype(np.float32), __meta__=np.array(json.dumps(meta)))
    n = sum(v.size for v in state.values())
    print(f'wrote {args.out}: {n:,} parameters, {Path(args.out).stat().st_size / 1e6:.1f} MB, prompt {meta["prompt"]}, '
          f'team {meta["team"]}', flush=True)
    if not args.replay:
        return 0

    sys.path.insert(0, str(RUNTIME))
    from kad import kad_numpy as K
    assert K.ORDERS == P.ORDERS and K.JOBS == P.JOBS, 'vocabularies differ'
    exact = K.KadModel({k: v.astype(np.float32) for k, v in state.items()}, meta['config'],
                       bias.numpy().astype(np.float32))
    half, _ = K.KadModel.load(args.out)
    worst = {'exact_order': 0.0, 'exact_job': 0.0, 'half_order': 0.0, 'half_job': 0.0}
    agree = {'exact_orders': 0, 'exact_jobs': 0, 'half_orders': 0, 'half_jobs': 0, 'positions': 0,
             'rows_equal': 0, 'land_first': 0}
    with zipfile.ZipFile(args.replay) as z:
        names = sorted(n for n in z.namelist() if n.endswith('.json'))[:args.games]
        for name in names:
            doc = json.loads(z.read(name))
            steps = doc['steps']
            arrays = AD.episode_arrays(doc)
            for seat in (0, 1):
                X = arrays[seat][0]
                turns = np.unique(np.linspace(0, len(X) - 1, args.turns).astype(int))
                advisor = K.KadAdvisor(Path(args.out))
                for t in range(len(X)):
                    view = AD.merge_observation(steps[t][0].get('observation') or {},
                                                steps[t][seat].get('observation') or {}, seat, t)
                    advisor.see(view)
                    if t not in turns:
                        continue
                    agree['rows_equal'] += int(np.array_equal(advisor.rows[-1], X[t]))
                    batch_np = K.inputs_at(X, t)
                    batch_t = {k: torch.as_tensor(v)[None] for k, v in P.inputs_at(X, t, c.history).items()}
                    with torch.no_grad():
                        h, pad = model.encode(batch_t, player.condition, player.team)
                        _, units = model.split(h)
                        job_t = torch.softmax(model.job(units), -1)[0].numpy()
                        orders = torch.zeros(1, P.MARKET_SLOTS, dtype=torch.long)
                        amounts = torch.zeros_like(orders)
                        probs_t = np.zeros((P.MARKET_SLOTS, len(P.ORDERS)), dtype=np.float32)
                        for k in range(P.MARKET_SLOTS):
                            d = model.market_state(h, pad, orders, amounts)
                            logits = model.order(d)[:, k]
                            probs_t[k] = torch.softmax(logits, -1)[0].numpy()
                            orders[:, k] = logits.argmax(-1)
                            amounts[:, k] = model.order_amount(d + model.order_embed(orders))[:, k].argmax(-1)
                    present = batch_np['units_mask'].astype(bool)
                    for tag, m in (('exact', exact), ('half', half)):
                        hh, pp = m.encode(batch_np)
                        o, a, pr = m.market(hh, pp)
                        jb = m.jobs(hh)
                        worst[f'{tag}_order'] = max(worst[f'{tag}_order'], float(np.abs(pr - probs_t).max()))
                        worst[f'{tag}_job'] = max(worst[f'{tag}_job'],
                                                  float(np.abs(jb[present] - job_t[present]).max()) if present.any() else 0.0)
                        agree[f'{tag}_orders'] += int(np.array_equal(o, orders[0].numpy())
                                                      and np.array_equal(a, amounts[0].numpy()))
                        agree[f'{tag}_jobs'] += int(np.array_equal(jb[present].argmax(-1), job_t[present].argmax(-1)))
                    agree['positions'] += 1
                    agree['land_first'] += int(P.ORDERS[int(orders[0, 0])][0] == 'BUY_LAND')
                print(f'{name} seat {seat}: {agree["positions"]} positions so far', flush=True)
    print(json.dumps({'worst_abs_prob_diff': worst, 'agreement': agree}), flush=True)
    advisor = K.KadAdvisor(Path(args.out))
    with zipfile.ZipFile(args.replay) as z:
        doc = json.loads(z.read(names[0]))
    steps = doc['steps']
    times = []
    for t in range(0, 300):
        advisor.see(AD.merge_observation(steps[t][0].get('observation') or {}, steps[t][0].get('observation') or {},
                                         0, t))
        if t % 25 == 24:
            started = time.perf_counter()
            advice = advisor.advise()
            times.append((time.perf_counter() - started) * 1000)
    print(f'advise on one thread: {np.mean(times[1:]):.0f} ms mean, first call {times[0]:.0f} ms (loads the weights); '
          f'last advice orders {advice["orders"][:3]}, P(BUY_LAND) {advisor.probability(advice, "BUY_LAND"):.3f}',
          flush=True)
    ok = (agree['exact_orders'] == agree['positions'] and agree['exact_jobs'] == agree['positions']
          and agree['rows_equal'] == agree['positions'] and worst['exact_order'] < 1e-3)
    print('KAD_EXPORT_CHECK ' + ('PASS' if ok else 'FAIL'), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
