#!/usr/bin/env python3
"""Compare native adapter paths with retained route-oracle logits.

Generate adapter logits with route-oracle and --lora-scaled FILE:1.75, Q8 K/V,
40 CPU MoE blocks and ROUTE_FORCE pointing to the same .tokens fixture. Generate
the base oracle with the same command minus the adapter. This catches the CPU
versus CUDA adapter arithmetic regression without relying on generated text.
"""
import argparse,json,os,struct,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for k in ('cli','model','adapter','tokens','oracle','base-oracle','profile','out'):p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--gpu',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
_,n=struct.unpack('<II',a.tokens.read_bytes()[:8]);assert n>1
rows={}
for mode in ('plain','pool','adaptive','off','zero'):
    opts=[]
    if mode!='off':opts+=['--lora-scaled',str(a.adapter)+(':0' if mode=='zero' else ':1.75')]
    if mode!='plain':opts+=['--pool','--pool-threads','16']
    if mode=='adaptive':opts+=['--cache-slots','-1','--profile',str(a.profile),'--adapt','--async-swaps','--overlap','--prefill-ubatch','auto','--lend','--kv-initial','32768','--dcheck-layer','18']
    cmd=[str(a.cli),'-m',str(a.model),'--tokens',str(a.tokens),'--oracle',str(a.base_oracle if mode in ('off','zero') else a.oracle),'-c','131072','-ctk','q8_0','-ctv','q8_0','-b','512','-ub','512','-t','16','--n-cpu-moe','40',*opts]
    r=subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID'),capture_output=True,text=True,check=True)
    (a.out/(mode+'.log')).write_text(r.stderr);row=json.loads(r.stdout);rows[mode]=row
    (a.out/(mode+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    assert row['steps_compared']==n,(mode,'incomplete oracle')
    if mode=='adaptive':assert row['dcheck_rows']>0 and row['dcheck_bad']==0,(mode,'expert row mismatch')
    else:assert row['logits_bit_identical']==n,(mode,'logit mismatch')
    print(mode,'PASS',flush=True)
(a.out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
