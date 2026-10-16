#!/usr/bin/env python3
"""Reproduce the LFM numerical stages and public calibration (one visible NVIDIA GPU).

python3 lite/tests/lfm25_validate.py --bin build-lite/bin --model MODEL.gguf --out /path/to/experiment
Raw logits, routes and generated text stay in --out. The summary is numeric; do not publish raw traces.
"""
import argparse, hashlib, json, os, subprocess
from pathlib import Path

ap=argparse.ArgumentParser(__doc__)
ap.add_argument('--bin',type=Path,required=True)
ap.add_argument('--model',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--gpu',help='CUDA device UUID; omit to use the existing CUDA_VISIBLE_DEVICES')
a=ap.parse_args();a.bin=a.bin.resolve();a.model=a.model.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[1]
for name in ['math','code','science','analysis','heldout']:
 assert (root/'tests/lfm25'/f'{name}.json').is_file(), f'missing public input: {name}'
env=os.environ.copy()
if a.gpu:env.update(CUDA_VISIBLE_DEVICES=a.gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID')
h=hashlib.sha256()
with a.model.open('rb') as f:
 for block in iter(lambda:f.read(8<<20),b''):h.update(block)
assert h.hexdigest()=='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5','wrong checkpoint'

def run(name,args,extra=None):
 print(name, flush=True)
 with (a.out/(name+'.stdout')).open('w') as out,(a.out/(name+'.stderr')).open('w') as err:
  subprocess.run([str(x) for x in args],env=dict(env,**(extra or {})),stdout=out,stderr=err,check=True)

common=['-m',a.model,'-ngl','99','--n-cpu-moe','40','-c','4096','-b','512','-ub','512','-fa','on','-ctk','f16','-ctv','f16','-t','16','--temp','0','--seed','42']
names=['math','code','science','analysis','heldout']
for name in names:
 prefix=a.out/(name+'-oracle')
 run(name+'-oracle',[a.bin/'route-oracle',*common],{'ROUTE_REQ':str(root/'tests/lfm25'/f'{name}.json'),'ROUTE_OUT':str(prefix),'ROUTE_NGEN':'192','ROUTE_LOGITS':'1'})
 if name!='heldout':
  # Same greedy route recorder and public input as the published four-prompt seed.
  run(name+'-routes',[a.bin/'route-log',*common],{'ROUTE_REQ':str(root/'tests/lfm25'/f'{name}.json'),'ROUTE_OUT':str(a.out/(name+'-routes')),'ROUTE_NGEN':'192'})
profile=a.out/'profile.txt'
run('profile',['python3',root/'tools/g4_profile.py',profile,*[a.out/(n+'-routes.routes') for n in names if n!='heldout']],{'PROFILE_NE':'64'})
summary={'model_sha256':h.hexdigest(),'llama_commit':'b6b003d2cb29647d968302eb2db8da6f66303b3e','cases':{}}
for name in names:
 summary['cases'][name]={}
 for mode in ['plain','pool','cache','adaptive']:
  opts=[] if mode=='plain' else ['--pool','--pool-threads','16']
  if mode in ('cache','adaptive'):opts+=['--cache-slots','40','--profile',profile,'--dcheck-layer','39']
  if mode=='adaptive':opts+=['--adapt','--async-swaps','--overlap','--prefill-ubatch','512','--lend']
  key=name+'-'+mode
  run(key,[a.bin/'strata-qwen35','-m',a.model,'--tokens',a.out/(name+'-oracle.tokens'),'--oracle',a.out/(name+'-oracle.logits'),'-c','8192','-b','512','-ub','512','-t','16','--n-cpu-moe','40',*opts])
  j=json.loads((a.out/(key+'.stdout')).read_text());summary['cases'][name][mode]=j
  assert j['steps_compared']==192,(key,'incomplete oracle')
  if mode in ('plain','pool'):assert j['logits_bit_identical']==192,(key,'numerical regression')
  if mode in ('cache','adaptive'):assert j['dcheck_rows']>0 and j['dcheck_bad']==0,(key,'expert row mismatch')
run('state',[a.bin/'lfm25-state',a.model,a.out/'math-oracle.tokens'])
summary['state']=[json.loads(line) for line in (a.out/'state.stdout').read_text().splitlines()]
(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(a.out/'summary.json')
