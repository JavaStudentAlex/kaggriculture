import json, os, time, zipfile, csv, io
from concurrent.futures import ThreadPoolExecutor
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
SUB = 56193386; OUT = "/results/kagg/our_replays/orchard_tide"
LB = "/tmp/claude-1317/-home-jovyan-kaggriculture/6bf614b1-d76e-495f-a6bc-0215880a28a5/scratchpad/lb/kaggriculture.zip"
z = zipfile.ZipFile(LB); lb = {int(r['TeamId']): float(r['Score']) for r in csv.DictReader(io.StringIO(z.read(z.namelist()[0]).decode()))}
games = []
for e in api.competition_list_episodes(SUB):
    e = e.to_dict()
    if e['state'] != 'COMPLETED': continue
    me = [a for a in e['agents'] if a['submissionId'] == SUB][0]; ot = [a for a in e['agents'] if a['submissionId'] != SUB]
    if not ot or me.get('reward') is None or ot[0].get('reward') is None: continue
    games.append({"id": e['id'], "time": e['createTime'], "seat": me.get('index', 0), "me": me['reward'], "opp": ot[0]['reward'],
                  "opp_team": ot[0]['teamName'], "opp_rating": lb.get(ot[0]['teamId']), "won": me['reward'] > ot[0]['reward']})
json.dump(games, open(f"{OUT}/index.json", "w"), indent=1)
print("games", len(games), "losses", sum(not g['won'] for g in games))
def fetch(g):
    p = f"{OUT}/{g['id']}.json"
    if os.path.exists(p) and os.path.getsize(p) > 1e6: return "have"
    for attempt in range(4):
        try:
            api.competition_episode_replay(g['id'], OUT)   # writes <id>.json
            if os.path.exists(p): return "ok"
            # some SDK versions name differently
            cands = [f for f in os.listdir(OUT) if str(g['id']) in f]
            return "ok?" + ",".join(cands)
        except Exception as ex:
            if "429" in str(getattr(ex, 'status', '')) or "429" in str(ex)[:40]:
                time.sleep(60 * (attempt + 1)); continue
            return f"err {type(ex).__name__}: {str(ex)[:80]}"
    return "throttled"
t0 = time.time()
with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(fetch, games))
import collections; print(collections.Counter(r.split(':')[0] for r in res), f"{time.time()-t0:.0f}s")
print([r for r in res if r not in ('ok','have')][:5])
