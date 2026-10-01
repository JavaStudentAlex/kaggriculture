"""Steps of chain_kad_run.sh that need Python (2026-09-29; cliproxyapi).

    python chain_helpers.py refit TRAINER_STATE.json          -> prints "epoch0 best best_epoch" (held-out AUC)
    python chain_helpers.py levers KAD_EXPERIMENT.json        -> prints "_KC_SELL=true _KC_HANDS=false" style pairs
    python chain_helpers.py provenance SEED.json CHECKPOINT_DIR NAME   (records the installed predictor in the seed)
"""
import hashlib
import json
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refit(state_path):
    history = json.loads(Path(state_path).read_text()).get('log_history') or []
    evals = [(h.get('epoch', 0.0), h['eval_auc_any_sell']) for h in history if 'eval_auc_any_sell' in h]
    epoch0 = next((a for e, a in evals if e == 0), evals[0][1] if evals else float('nan'))
    trained = [(a, e) for e, a in evals if e > 0]
    best, best_epoch = max(trained) if trained else (float('nan'), 0)
    print(f'{epoch0:.4f} {best:.4f} {best_epoch:g}')


def levers(path):
    """A lever stays on when its games gained results (or tied on results and did not lose money), else it starts
    off: KAD is then advice only (info['kad'] for the tactic stage; user, 2026-09-29), and the evolution may switch a
    lever on where its games pay."""
    summary = json.loads(Path(path).read_text())['summary']
    score = {}
    for lever in ('sell', 'hands'):
        s = summary.get(lever) or {}
        net = s.get('results_up', 0) - s.get('results_down', 0)
        score[lever] = (net, s.get('mean', 0.0))
    on = {lever: (net > 0 or (net == 0 and mean >= 0)) for lever, (net, mean) in score.items()}
    print(f"_KC_SELL={'true' if on['sell'] else 'false'} _KC_HANDS={'true' if on['hands'] else 'false'}")
    print(f'# sell {score["sell"]}, hands {score["hands"]} (results net, mean $/game)', file=sys.stderr)


def provenance(seed_path, checkpoint, name):
    seed = json.loads(Path(seed_path).read_text())
    ck = Path(checkpoint)
    pins = seed['provenance']['runtime_bundle_hashes']
    for f in ('config.json', 'labels.json', 'model.safetensors', 'scaler.npz', 'calibration.json'):
        if (ck / f).exists():
            pins[f'checkpoint/{f}'] = sha(ck / f)
    seed.setdefault('source', {})['oracle_model_sha256'] = sha(ck / 'model.safetensors')
    info = {'name': name, 'source': f'{name}: the 2026-09-28 refit (days 09-22..25, 27, 28), colab2 L4, '
                                    'installed by kad_tools/chain_kad_run.sh',
            'model_sha256': sha(ck / 'model.safetensors'), 'scaler_sha256': sha(ck / 'scaler.npz'),
            'config_sha256': sha(ck / 'config.json'), 'labels_sha256': sha(ck / 'labels.json')}
    if (ck / 'calibration.json').exists():
        info['calibration'] = {'sha256': sha(ck / 'calibration.json'), 'fitted_on': 'calibration/game_set (own games)'}
    seed['model_checkpoint'] = info
    Path(seed_path).write_text(json.dumps(seed, indent=2) + '\n')
    print(f'{seed_path}: predictor {name} ({info["model_sha256"][:12]}), calibration '
          f'{"yes" if "calibration" in info else "no"}')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'refit':
        refit(sys.argv[2])
    elif cmd == 'levers':
        levers(sys.argv[2])
    elif cmd == 'provenance':
        provenance(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(f'unknown command {cmd}')
