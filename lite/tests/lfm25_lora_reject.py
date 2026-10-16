#!/usr/bin/env python3
"""Reject incompatible adapters through the real CLI loader. Uses tiny synthetic GGUF fixtures.
The token fixture may be any valid route-oracle .tokens file; rejected cases never run inference.
"""
import argparse,json,os,struct,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for name in ('cli','model','adapter','tokens','out'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--gpu',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
expected={'architecture':'expected an lfm2moe LoRA adapter','rank':'only f32 rank-one','shape':'LoRA/base shape mismatch','target':'only LFM mixer-output and expert-down','finite':'non-finite LoRA tensor','scale':'LoRA scale must be finite'}
rows=[]
for name,error in expected.items():
    path=a.out/(name+'.gguf')
    if name=='scale':path=a.adapter
    else:
        def string(v):
            b=v.encode();return struct.pack('<Q',len(b))+b
        kv=[('general.architecture','qwen35moe' if name=='architecture' else 'lfm2moe'),('general.type','adapter'),('adapter.type','lora')]
        header=b'GGUF'+struct.pack('<IQQ',3,2,4)
        for k,v in kv:header+=string(k)+struct.pack('<I',8)+string(v)
        header+=string('adapter.lora_alpha')+struct.pack('<If',6,1.)
        target='blk.14.attn_q.weight' if name=='target' else 'blk.14.attn_output.weight'
        rank=2 if name=='rank' else 1
        dims=[(2047 if name=='shape' else 2048,rank),(rank,2048)]
        data=b''
        for suffix,(x,y) in zip(('.lora_a','.lora_b'),dims):
            header+=string(target+suffix)+struct.pack('<IQQIQ',2,x,y,0,len(data))
            tensor=bytes(4*x*y)
            if name=='finite' and suffix=='.lora_a':tensor=struct.pack('<f',float('nan'))+tensor[4:]
            data+=tensor;data+=bytes((-len(data))%32)
        header+=bytes((-len(header))%32);path.write_bytes(header+data)
    cmd=[str(a.cli),'-m',str(a.model),'--tokens',str(a.tokens),'--lora',str(path),'--lora-scale','nan' if name=='scale' else '1.75','-c','2048','--n-cpu-moe','40']
    r=subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu),capture_output=True,text=True)
    (a.out/(name+'.stderr')).write_text(r.stderr)
    row={'case':name,'exit_code':r.returncode,'expected':error,'passed':r.returncode==1 and error in r.stderr};rows.append(row)
    print(row,flush=True)
    assert row['passed'],r.stderr[-2000:]
(a.out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
