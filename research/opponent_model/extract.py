"""Stream Kaggriculture replays into a training table for opponent supply.

Label (exact, from inventory accounting):

    total_supply[p](t) = inv[t+1][p] - inv[t][p] + town_draw[p](t)

because within a step the engine runs _apply_unit_action -> _process_market ->
_town_consume (engine:941-947). total_supply is both players' executed net flow.
Attribution to the opponent is exact whenever only one seat requested that
product; rows where both did are flagged `contested` and split pro-rata.

Opponent `private` from the replay is written to a separate diagnostics array
for validating a shed-belief filter. It is never a feature.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mechanics import PRODUCTS, config_intervals, parse_market_orders, town_draw

from features import OpponentHistory, build_features, feature_names


def episode_rows(doc, feat_names, stride=1):  # noqa: C901 -- one pass over the replay
    """Yield (features, labels, diagnostics) per seat per step for one episode."""
    steps = doc.get("steps") or []
    if len(steps) < 2:
        return
    cfg = config_intervals(doc.get("configuration"))
    max_orders = int((doc.get("configuration") or {}).get("maxMarketOrdersPerTurn", 10) or 10)
    episode_id = int((doc.get("info") or {}).get("EpisodeId") or 0)
    histories = [OpponentHistory(), OpponentHistory()]

    for t in range(len(steps) - 1):
        cur, nxt = steps[t], steps[t + 1]
        obs0, nxt0 = cur[0].get("observation") or {}, nxt[0].get("observation") or {}
        market, next_market = obs0.get("market") or {}, nxt0.get("market") or {}
        inv, next_inv = market.get("inventory") or {}, next_market.get("inventory") or {}
        if not inv or not next_inv or not obs0.get("farms"):
            continue

        shops = (obs0.get("town") or {}).get("unlocked_shops") or []
        drawn = town_draw(t, shops, cfg["shop_interval"], cfg["center_interval"])

        # Both seats' requested orders; the engine truncates at maxMarketOrders.
        requested = []
        for seat in (0, 1):
            action = cur[seat].get("action") if seat < len(cur) else None
            action = dict(action or {})
            action["market"] = (action.get("market") or [])[:max_orders]
            requested.append(parse_market_orders(action))

        total_supply, contested, per_seat_supply = Counter(), {}, [Counter(), Counter()]
        for p in PRODUCTS:
            delta = int(next_inv.get(p, 0)) - int(inv.get(p, 0))
            total = delta + drawn.get(p, 0)
            total_supply[p] = total
            asked = [requested[s][0].get(p, 0) - requested[s][1].get(p, 0) for s in (0, 1)]
            active = [s for s in (0, 1) if asked[s] != 0]
            if len(active) == 1:
                per_seat_supply[active[0]][p] = total
            elif len(active) == 2:
                contested[p] = True
                denom = sum(abs(a) for a in asked) or 1
                for s in (0, 1):
                    per_seat_supply[s][p] = round(total * abs(asked[s]) / denom)

        for me in (0, 1):
            opp = 1 - me
            opp_supply = per_seat_supply[opp]

            if t % stride:
                # History must still advance on skipped steps; see below.
                histories[me].update(t, opp_supply)
                continue
            seat_obs = (cur[me].get("observation") or {}) if me < len(cur) else {}
            # Seat 1 replays omit `step` (evaluation-fidelity.md); restore it.
            obs = dict(obs0)
            obs["private"] = seat_obs.get("private") or {}
            obs["step"] = t
            obs["player"] = me

            feats = build_features(obs, me, cfg, histories[me], cfg["shed_capacity"])
            x = np.array([feats[k] for k in feat_names], dtype=np.float32)
            y = np.array([opp_supply.get(p, 0) for p in PRODUCTS], dtype=np.int32)

            opp_private = (cur[opp].get("observation") or {}).get("private") or {}
            opp_shed = opp_private.get("shed") or {}
            diag = np.array(
                [episode_id, t, me, int(any(contested.values()))]
                + [int(opp_shed.get(p, 0) or 0) for p in PRODUCTS]
                + [total_supply.get(p, 0) for p in PRODUCTS],
                dtype=np.int32,
            )
            yield x, y, diag
            # Update AFTER featurising: features at step t may only reflect
            # opponent sells up to t-1, which is what a live agent can know.
            histories[me].update(t, opp_supply)


def run(archives, out_dir, max_episodes=None, stride=1):
    feat_names = feature_names()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grand = 0

    for archive in archives:
        xs, ys, ds = [], [], []
        episodes = 0
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if m.endswith(".json")]
            if max_episodes:
                members = members[:max_episodes]
            for member in members:
                try:
                    doc = json.loads(zf.read(member))
                except (json.JSONDecodeError, KeyError):
                    continue
                for x, y, d in episode_rows(doc, feat_names, stride):
                    xs.append(x)
                    ys.append(y)
                    ds.append(d)
                episodes += 1

        if not xs:
            continue
        dest = out_dir / (Path(archive).stem + ".npz")
        np.savez_compressed(
            dest,
            X=np.stack(xs),
            Y=np.stack(ys),
            D=np.stack(ds),
            feature_names=np.array(feat_names),
            products=np.array(PRODUCTS),
        )
        grand += len(xs)
        print(f"{Path(archive).name}: {episodes} episodes -> {len(xs):,} rows -> {dest.name}", flush=True)

    print(f"\ntotal rows: {grand:,}")
    return grand


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archives", nargs="+", help="replay .zip archives")
    ap.add_argument("--out", required=True, help="output directory for .npz shards")
    ap.add_argument("--max-episodes", type=int, default=None, help="cap episodes per archive")
    ap.add_argument("--stride", type=int, default=1, help="emit every Nth step")
    args = ap.parse_args()
    run(args.archives, args.out, args.max_episodes, args.stride)


if __name__ == "__main__":
    main()
