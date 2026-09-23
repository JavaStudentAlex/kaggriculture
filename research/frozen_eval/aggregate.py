"""Validate all five remote result sets before reporting a frozen benchmark."""
from pathlib import Path
import collections,json,sys
import runner
HERE=Path(__file__).resolve().parent
RUN=HERE/'runs/iter27_20260923'
manifest=json.loads((RUN/'payload/manifest.json').read_text())
expected={j['job_id']:j for i in range(5) for j in runner.build_jobs(manifest,i,5)}
records={};fingerprints=set();shards=[]
for i in range(5):
    root=RUN/'remote_results'/str(i)/'results';summary=root/'summary.json';rows=root/'results.jsonl'
    if not summary.exists() or not rows.exists():
        shards.append({'shard':i,'state':'missing'});continue
    s=json.loads(summary.read_text());assert s['evaluation_id']==manifest['evaluation_id']
    fingerprints.add(s['source_fingerprint'])
    shard_expected={j['job_id']:j for j in runner.build_jobs(manifest,i,5)}
    loaded=runner.load_results(rows,list(shard_expected.values()),s['source_fingerprint'])
    for jid,row in loaded.items():
        assert jid not in records,'Duplicate across shards'
        records[jid]=row
    shards.append({'shard':i,'state':s['status'],'completed':len(loaded),'planned':len(shard_expected)})
assert len(fingerprints)<=1,'Different deployed artifact/engine fingerprints across shards'
counts=collections.Counter(r['outcome'] for r in records.values())
external=[r for r in records.values() if r['opponent_kind']!='self_control']
competitive=collections.Counter(r['outcome'] for r in external)
per={}
for opponent in manifest['opponents']:
    per[opponent['id']]={'label':opponent['label'],'kind':opponent['kind'],'seats':{}}
    for seat in (0,1):
        selected=[r for r in records.values() if r['opponent_id']==opponent['id'] and r['seat']==seat]
        c=collections.Counter(r['outcome'] for r in selected)
        per[opponent['id']]['seats'][seat]={'completed':len(selected),'planned':100,'counts':dict(c)}
report={'evaluation_id':manifest['evaluation_id'],'complete':len(records)==len(expected),
        'coverage':{'completed':len(records),'planned':len(expected),'missing':len(expected)-len(records)},
        'valid_complete':len(records)==len(expected) and not counts['INVALID'],
        'all_counts':dict(counts),'competitive_counts':dict(competitive),
        'self_control_excluded_from_competitive':True,'shards':shards,'opponents':per}
(RUN/'aggregate.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='opponents'},indent=2))
