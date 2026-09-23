"""Read-only full-game preflight of staged restored opponents against a frozen graph."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import concurrent.futures
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'shinka/evolution'))
sys.path.insert(0,str(Path(__file__).resolve().parent))

def match(task):
    opponent,graph,seat,seed=task
    import evaluate
    from kaggle_environments import make
    started=time.time()
    try:
        a=evaluate.load_agent(graph,'preflight_candidate')
        b=evaluate.load_agent(opponent,'preflight_opponent')
        env=make('kaggriculture',configuration={'episodeSteps':720,'seed':seed})
        env.run([a,b] if seat==0 else [b,a])
        last=env.steps[-1]
        result=dict(opponent=Path(opponent).name,seat=seat,seed=seed,steps=len(env.steps),status=[s['status'] for s in last],rewards=[s['reward'] for s in last],seconds=round(time.time()-started,2))
        result['ok']=all(s=='DONE' for s in result['status']) and result['steps']>=720
    except Exception:
        result=dict(opponent=Path(opponent).name,seat=seat,seed=seed,ok=False,error=traceback.format_exc())
    print(json.dumps(result),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--pool',type=Path,required=True);p.add_argument('--graph',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=14);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    # Avoid importing the full evolutionary runner and its LLM/embedding stack.
    wrapper=a.output/'frozen_graph_wrapper.py'
    wrapper.write_text('import os,sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(ROOT/'research/procedural_graph'))+')\nos.environ["KAGG_GRAPH_PATH"]='+repr(str(a.graph.resolve()))+'\nimport agent_graph\nagent_graph._ENGINE=None\nagent=agent_graph.agent\n')
    opponents=sorted(a.pool.glob('champ_restored_*.py'));assert opponents
    tasks=[(str(f),str(wrapper),s,seed) for f in opponents for s,seed in [(0,101),(1,70102)]]
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as executor:
        fs=[executor.submit(match,t) for t in tasks]
        for f in concurrent.futures.as_completed(fs):
            results.append(f.result())
            (a.output/'progress.json').write_text(json.dumps(results,indent=2))
    ok=all(r['ok'] for r in results)
    report=dict(ok=ok,games=len(results),opponents=len(opponents),results=results)
    (a.output/'report.json').write_text(json.dumps(report,indent=2));print('VALIDATION_REPORT',json.dumps(report),flush=True)
    sys.exit(0 if ok else 1)
if __name__=='__main__':main()
