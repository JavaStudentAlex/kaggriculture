import json, glob, collections, statistics as st, sys
sys.path.insert(0, '/home/jovyan/kaggriculture/research/opponent_model')
from mechanics import town_draw, parse_market_orders, config_intervals, PRODUCTS
OUT = "/results/kagg/our_replays/orchard_tide"
index = {g['id']: g for g in json.load(open(f"{OUT}/index.json"))}
def game(path):
    d = json.load(open(path)); steps = d['steps']; eid = d['info']['EpisodeId']; g = index[eid]
    me = g['seat']; cfg = config_intervals(d.get('configuration'))
    res = {"id": eid, "won": g['won'], "seat": me, "opp_rating": g['opp_rating']}
    money = {s: [steps[t][0]['observation']['farms'][s]['money'] for t in range(720)] for s in (0,1)}
    inc = {s: collections.Counter() for s in (0,1)}; spend = {s: collections.Counter() for s in (0,1)}
    for s in (0,1):
        for t in range(719):
            dlt = money[s][t+1] - money[s][t]
            if dlt > 0: inc[s][t//24] += dlt
            else: spend[s][t//24] -= dlt
    # accounting-based per-seat supply (as extract.py)
    supply = {s: collections.Counter() for s in (0,1)}; rev = {s: collections.Counter() for s in (0,1)}; sup_day = {s: collections.Counter() for s in (0,1)}
    for t in range(719):
        obs0, nxt0 = steps[t][0]['observation'], steps[t+1][0]['observation']
        inv, ninv = obs0['market']['inventory'], nxt0['market']['inventory']; prices = obs0['market']['prices']
        shops = (obs0.get('town') or {}).get('unlocked_shops') or []
        drawn = town_draw(t, shops, cfg['shop_interval'], cfg['center_interval'])
        req = []
        for s in (0,1):
            a = dict(steps[t+1][s].get('action') or {}); a['market'] = (a.get('market') or [])[:10]
            req.append(parse_market_orders(a))
        for p in PRODUCTS:
            total = int(ninv.get(p,0)) - int(inv.get(p,0)) + drawn.get(p,0)
            asked = [req[s][0].get(p,0) - req[s][1].get(p,0) for s in (0,1)]
            act = [s for s in (0,1) if asked[s] != 0]
            if len(act) == 1: share = {act[0]: total}
            elif len(act) == 2:
                den = sum(abs(a) for a in asked) or 1; share = {s: round(total*abs(asked[s])/den) for s in (0,1)}
            else: share = {}
            for s, u in share.items():
                if u > 0: supply[s][p] += u; rev[s][p] += u*prices.get(p,0); sup_day[s][t//24] += u*prices.get(p,0)
    res.update(money_final=(money[me][719], money[1-me][719]), inc={s: dict(inc[s]) for s in (0,1)}, spend={s: dict(spend[s]) for s in (0,1)},
               supply={s: dict(supply[s]) for s in (0,1)}, rev={s: dict(rev[s]) for s in (0,1)}, sup_day={s: dict(sup_day[s]) for s in (0,1)})
    def farm(t, s):
        f = steps[t][0]['observation']['farms'][s]; tiles = [x for r in f['tiles'] for x in r if isinstance(x, dict)]
        return {"hands": len(f['hands']), "animals": sum(1 for x in tiles if x.get('kind')=='PASTURE' and x.get('animal')), "plants": sum(1 for x in tiles if x.get('kind')=='PLANT'),
                "anim": collections.Counter(x.get('animal') for x in tiles if x.get('kind')=='PASTURE' and x.get('animal'))}
    res['farm'] = {s: {dd: farm(dd*24+12, s) for dd in (2,5,8,11,14,17,20,23,26,29)} for s in (0,1)}
    # shed at end (unsold stock) and last-day dump
    res['shed_end'] = {s: sum(v for k, v in (steps[719][s]['observation'].get('private') or {}).get('shed', {}).items() if k in PRODUCTS) for s in (0,1)}
    return res
games = [game(p) for p in sorted(glob.glob(f"{OUT}/episode-*-replay.json"))]
json.dump(games, open(f"{OUT}/stats2.json", "w"))
W = [g for g in games if g['won']]; L = [g for g in games if not g['won']]
mean = lambda xs: st.mean(xs) if xs else float('nan')
def me(g): return g['seat']
print(f"wins {len(W)} losses {len(L)}")
print("\n== income (money-delta) per game-day, me vs opp")
for lab, G in (("wins", W), ("losses", L)):
    print(f"{lab:6} me ", " ".join(f"{mean([g['inc'][me(g)].get(dd,0) for g in G]):>6.0f}" for dd in range(30)))
    print(f"{lab:6} opp", " ".join(f"{mean([g['inc'][1-me(g)].get(dd,0) for g in G]):>6.0f}" for dd in range(30)))
print("\n== spend per game-day, me vs opp")
for lab, G in (("wins", W), ("losses", L)):
    print(f"{lab:6} me ", " ".join(f"{mean([g['spend'][me(g)].get(dd,0) for g in G]):>6.0f}" for dd in range(30)))
    print(f"{lab:6} opp", " ".join(f"{mean([g['spend'][1-me(g)].get(dd,0) for g in G]):>6.0f}" for dd in range(30)))
print("\n== totals: income, spend, final (me | opp)")
for lab, G in (("wins", W), ("losses", L)):
    print(f"{lab:6} income {mean([sum(g['inc'][me(g)].values()) for g in G]):,.0f}|{mean([sum(g['inc'][1-me(g)].values()) for g in G]):,.0f}  spend {mean([sum(g['spend'][me(g)].values()) for g in G]):,.0f}|{mean([sum(g['spend'][1-me(g)].values()) for g in G]):,.0f}  final {mean([g['money_final'][0] for g in G]):,.0f}|{mean([g['money_final'][1] for g in G]):,.0f}  unsold shed units at end {mean([g['shed_end'][me(g)] for g in G]):.0f}|{mean([g['shed_end'][1-me(g)] for g in G]):.0f}")
print("\n== units actually supplied per product (accounting), me | opp")
for lab, G in (("wins", W), ("losses", L)):
    print(lab, {p: f"{mean([g['supply'][me(g)].get(p,0) for g in G]):.0f}|{mean([g['supply'][1-me(g)].get(p,0) for g in G]):.0f}" for p in PRODUCTS})
print("\n== revenue per product (units x price seen), me | opp")
for lab, G in (("wins", W), ("losses", L)):
    print(lab, {p: f"{mean([g['rev'][me(g)].get(p,0) for g in G]):,.0f}|{mean([g['rev'][1-me(g)].get(p,0) for g in G]):,.0f}" for p in PRODUCTS})
print("\n== farm at hour 12 of day d: hands, animals, plants (me | opp)")
for lab, G in (("wins", W), ("losses", L)):
    print(lab, " ".join(f"d{dd}: h{mean([g['farm'][me(g)][dd]['hands'] for g in G]):.1f}|{mean([g['farm'][1-me(g)][dd]['hands'] for g in G]):.1f} a{mean([g['farm'][me(g)][dd]['animals'] for g in G]):.1f}|{mean([g['farm'][1-me(g)][dd]['animals'] for g in G]):.1f} p{mean([g['farm'][me(g)][dd]['plants'] for g in G]):.0f}|{mean([g['farm'][1-me(g)][dd]['plants'] for g in G]):.0f}" for dd in (2,5,8,11,14,17,20,23,26,29)))
print("\n== animal mix day 20 (me | opp)")
for lab, G in (("wins", W), ("losses", L)):
    am=collections.Counter(); ao=collections.Counter()
    for g in G: am.update(g['farm'][me(g)][20]['anim']); ao.update(g['farm'][1-me(g)][20]['anim'])
    print(lab, {k: round(v/len(G),1) for k,v in am.items()}, "|", {k: round(v/len(G),1) for k,v in ao.items()})
# loss margin distribution and correlation with opponent animals
print("\n== losses: margin buckets")
m = sorted(g['money_final'][1]-g['money_final'][0] for g in L); print("loss margins: min", m[0], "q1", m[len(m)//4], "median", m[len(m)//2], "q3", m[3*len(m)//4], "max", m[-1])
print("close losses (<5k):", sum(1 for x in m if x < 5000), " blowouts (>25k):", sum(1 for x in m if x > 25000))
