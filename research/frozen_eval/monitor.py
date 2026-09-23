"""Read-only monitor for the five frozen benchmark kernels; never submits jobs."""
from pathlib import Path
import argparse,json,subprocess,time

HERE=Path(__file__).resolve().parent
RUN=HERE/'runs/iter27_20260923'

def command(args,timeout=90):
    p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    return p.returncode,p.stdout+p.stderr

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');parser.add_argument('--interval',type=int,default=180);a=parser.parse_args()
    deadline=time.monotonic()+24*3600
    while time.monotonic()<deadline:
        states=[]
        for shard in range(5):
            ref=f'sunshinethroughfog/kagg-iter27-benchmark-{shard}'
            rc,msg=command(['kaggle','kernels','status',ref]);state={'shard':shard,'ref':ref,'status_output':msg.strip(),'returncode':rc}
            if 'COMPLETE' in msg or 'ERROR' in msg:
                out=RUN/'remote_results'/str(shard);out.mkdir(parents=True,exist_ok=True)
                orc,omsg=command(['kaggle','kernels','output',ref,'-p',str(out)],timeout=180)
                state['download_returncode']=orc
                summary=out/'results/summary.json'
                if summary.exists():
                    s=json.loads(summary.read_text());state['summary']={k:s.get(k) for k in ['evaluation_id','status','overall','source_fingerprint','selection']}
                else:state['download_note']=omsg.strip()
                command(['kaggle','kernels','delete','-y',ref],timeout=30)
            states.append(state)
        (RUN/'monitor_status.json').write_text(json.dumps(states,indent=2))
        print(json.dumps(states),flush=True)
        if a.once:return
        if all('COMPLETE' in x['status_output'] or 'ERROR' in x['status_output'] for x in states):
            print('ALL_REMOTE_KERNELS_TERMINAL; inspect per-shard artifacts and exact job coverage before reporting a final score.',flush=True)
            return
        time.sleep(a.interval)
    raise SystemExit('Monitoring time limit reached; jobs may still be running')

if __name__=='__main__':main()
