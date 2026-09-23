"""Full-length isolated-seat smoke test of the frozen payload."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import sys,json,time,argparse
from concurrent.futures import ProcessPoolExecutor

def match(task):
    root,opponent,seat,seed=task
    sys.path.insert(0,root)
    from pool_upgrade_bundle_agent import BundleAgent
    from kaggle_environments import make
    p=Path(root)/'agents'
    ids=['candidate_iter27',opponent] if seat==0 else [opponent,'candidate_iter27']
    agents=[BundleAgent(p/i) for i in ids]
    t=time.monotonic()
    try:
        for a in agents:a.start()
        def a0(obs,conf):return agents[0](obs,conf)
        def a1(obs,conf):return agents[1](obs,conf)
        env=make('kaggriculture',configuration={'episodeSteps':720,'seed':seed},debug=True)
        env.run([a0,a1])
        last=env.steps[-1]
        out={'opponent':opponent,'seat':seat,'seed':seed,'steps':len(env.steps),'rewards':[s.reward for s in last],'statuses':[s.status for s in last], 'seconds':round(time.monotonic()-t,2)}
        out['valid']=out['steps']==720 and out['statuses']==['DONE','DONE']
        return out
    except Exception as exc:return {'opponent':opponent,'seat':seat,'error':str(exc),'valid':False}
    finally:
        for a in agents:a.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True);a=parser.parse_args()
    root=str(Path(a.root).resolve())
    tasks=[(root,n,seat,1900000027+seat) for n in ['submission_hazel_weir','submission_copper_weir','champ_evo_0027_Island-Watering_5abf926b8165'] for seat in [0,1]]
    records=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for out in pool.map(match,tasks):
            records.append(out);print('SMOKE_RESULT',json.dumps(out),flush=True)
            Path(a.output).write_text(json.dumps(records,indent=2))
    raise SystemExit(0 if all(r['valid'] for r in records) else 1)
