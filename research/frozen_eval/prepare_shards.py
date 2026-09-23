"""Generate private Kaggle CPU shard scripts; does not launch them."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
RUN=HERE/'runs/iter27_20260923'
base=(RUN/'preflight/run.py').read_text()
base=base[:base.index("subprocess.run([sys.executable,str(root/'smoke.py')")]
for shard in range(5):
    d=RUN/'shards'/str(shard);d.mkdir(parents=True,exist_ok=True)
    extra=f'''
SHARD={shard}
# All-opponent first-seed gate before the bulk run. These pilot games are not
# pooled into the final score and are kept in a separate output directory.
common=[sys.executable,str(root/'runner.py'),'--manifest',str(root/'manifest.json'),
        '--root',str(root),'--shard-index',str(SHARD),'--shards','5','--workers','4']
preflight=work/'pilot'
subprocess.run(common+['--output',str(preflight),'--smoke','--deadline-seconds','3600'],check=True)
pilot=json.loads((preflight/'summary.json').read_text())
if pilot.get('status') != 'complete' or pilot['overall']['invalid']:
    raise RuntimeError('Pilot did not complete cleanly; bulk dispatch blocked')
print('SHARD_PILOT_PASSED',SHARD,flush=True)
subprocess.run(common+['--output',str(work/'results'),'--seed-count','100',
                       '--deadline-seconds','32400'],check=True)
print('SHARD_FINISHED',SHARD,EVALUATION_ID,flush=True)
'''
    (d/'run.py').write_text(base+extra)
    meta={'id':f'sunshinethroughfog/kagg-iter27-benchmark-{{shard}}'.format(shard=shard),
          'title':f'kagg-iter27-benchmark-{shard}','code_file':'run.py','language':'python',
          'kernel_type':'script','is_private':True,'enable_gpu':False,'enable_tpu':False,
          'enable_internet':True,'dataset_sources':['sunshinethroughfog/kagg-frozen-iter27-20260923'],
          'competition_sources':['kaggriculture']}
    (d/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
print('Prepared five CPU shard scripts, each 30 pilot + 600 scored games')
