#!/usr/bin/env python3
"""Matched native-server LFM benchmark: f16/q8_0 K+V, cold prefill and cold/warm decode.

Both engines consume the same token prompts and generate the same count of greedy tokens
(ignore EOS). Contents can differ: this measures serving, not forced-token logit comparison.
The output preserves every command, fit attempt, numeric receipt and server log.
"""
import argparse,datetime,hashlib,json,os,socket,struct,subprocess,threading,time,urllib.error,urllib.request
from pathlib import Path

ap=argparse.ArgumentParser(__doc__)
for name in ('strata','llama','model','profile','out'):ap.add_argument('--'+name,type=Path,required=True)
ap.add_argument('--strata-commit',default='unspecified');ap.add_argument('--llama-commit',default='unspecified')
ap.add_argument('--gpu',required=True);ap.add_argument('--prompts',required=True,help='label=path.tokens,...')
ap.add_argument('--port',type=int,default=18473);ap.add_argument('--repetitions',type=int,default=2)
ap.add_argument('--ctx',type=int,default=131072);ap.add_argument('--generate',type=int,default=256)
a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID')
base=f'http://127.0.0.1:{a.port}'

def save(p,j):p.write_text(json.dumps(j,indent=2)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def get(path):return json.load(urllib.request.urlopen(base+path,timeout=10))
def post(body):
 req=urllib.request.Request(base+'/completion',json.dumps(body).encode(),{'Content-Type':'application/json'})
 t=time.monotonic()
 with urllib.request.urlopen(req,timeout=2400) as r:j=json.load(r)
 tm=j['timings'];assert tm['predicted_n']==body['n_predict'],('incomplete generation',tm)
 return {'wall_s':time.monotonic()-t,'timings':tm,'output_sha256':hashlib.sha256(j.get('content','').encode()).hexdigest()}
def vram():
 s=subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
 return int(s.strip())
class Peak(threading.Thread):
 def __init__(self):super().__init__(daemon=True);self.stop_event=threading.Event();self.peak=vram()
 def run(self):
  while not self.stop_event.wait(.5):self.peak=max(self.peak,vram())
 def finish(self):self.stop_event.set();self.join();return max(self.peak,vram())

def stop(proc,log):
 if proc.poll() is None:
  proc.terminate()
  try:proc.wait(timeout=30)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
 log.close()

def start(cmd,path):
 log=path.open('w');proc=subprocess.Popen(list(map(str,cmd)),env=env,stdout=log,stderr=log)
 deadline=time.monotonic()+180
 while proc.poll() is None and time.monotonic()<deadline:
  try:
   if get('/health').get('status')=='ok':return proc,log
  except (OSError,ValueError):pass
  time.sleep(.5)
 stop(proc,log);return None,None

prompts=[]
for spec in a.prompts.split(','):
 label,path=spec.split('=',1);p=Path(path);b=p.read_bytes();n,g=struct.unpack_from('<II',b)
 ids=list(struct.unpack_from('<'+'I'*n,b,8));assert n+a.generate<a.ctx
 prompts.append((label,ids,digest(p)))
# Full-context memory is exercised first; all cold requests explicitly disallow prompt reuse.
prompts.sort(key=lambda v:len(v[1]),reverse=True)
manifest={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_uuid':a.gpu,'ctx':a.ctx,'generated':a.generate,'repetitions':a.repetitions,'threads':16,'engine_commit':a.strata_commit,'llama_commit':a.llama_commit,'binaries':{str(p):digest(p) for p in (a.strata,a.llama)},'profile_sha256':digest(a.profile),'prompts':{n:{'tokens':len(ids),'sha256':h} for n,ids,h in prompts},'cells':{}}
if (a.out/'manifest.json').exists():
 old=json.loads((a.out/'manifest.json').read_text());manifest['started_utc']=old['started_utc'];manifest['resumed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
save(a.out/'manifest.json',manifest)
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',a.port))!=0,'benchmark port occupied'
for ikv,kv in enumerate(('f16','q8_0')):
 for iengine,engine in enumerate(('llama','strata')):
  key=engine+'-'+kv;dst=a.out/key;dst.mkdir(exist_ok=True)
  completed=dst/'result.json'
  if completed.exists():manifest['cells'][key]=json.loads(completed.read_text());continue
  basecmd=[a.llama if engine=='llama' else a.strata,'-m',a.model,'--host','127.0.0.1','--port',str(a.port),'--alias','lfm-bench','-c',str(a.ctx),'-t','16','-ctk',kv,'-ctv',kv]
  if engine=='llama':basecmd+=['-ngl','99','-fa','on','--parallel','1','-b','2048','-ub','512','--fit','off','--no-context-shift','--jinja','--no-webui','--timeout','2400']
  else:basecmd+=['--pool-threads','16','--profile',a.profile,'--kv-initial','32768','--prefill-ubatch','auto']
  attempts=json.loads((dst/'fit.json').read_text()) if (dst/'fit.json').exists() else []
  for ncm in (range(8,25) if engine=='llama' else [40]):
   port=a.port+(ikv*2+iengine)*100+ncm
   base=f'http://127.0.0.1:{port}'
   with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0,f'benchmark port {port} occupied'
   cmd=basecmd+['--n-cpu-moe',str(ncm)];cmd=list(map(str,cmd));cmd[cmd.index('--port')+1]=str(port)
   logpath=dst/f'server-ncm{ncm}-p{port}.log'
   print(f'{key} start n_cpu_moe={ncm}',flush=True)
   proc,log=start(cmd,logpath)
   attempt={'n_cpu_moe':ncm,'command':cmd,'loaded':proc is not None};attempts.append(attempt);save(dst/'fit.json',attempts)
   if proc is None:
    tail=(logpath).read_text(errors='replace')[-12000:].lower()
    if not any(s in tail for s in ('out of memory','failed to allocate','unable to allocate','cannot allocate','cuda error:')):raise RuntimeError(f'{key}: non-memory startup failure, see log')
    continue
   result={'engine':engine,'kv_k':kv,'kv_v':kv,'n_cpu_moe':ncm,'command':cmd,'requests':[],'vram_loaded_mib':vram()}
   try:
    # Equal untimed warm-up for CUDA initialization, then a full uncached prompt read.
    post({'prompt':prompts[-1][1][:64],'n_predict':8,'temperature':0,'top_k':1,'repeat_penalty':1,'ignore_eos':True,'cache_prompt':False})
    for rep in range(a.repetitions):
     for label,ids,_ in prompts:
      pair={'label':label,'prompt_tokens':len(ids),'repetition':rep+1};peak=Peak();peak.start()
      try:
       body={'prompt':ids,'n_predict':a.generate,'temperature':0,'top_k':1,'repeat_penalty':1,'ignore_eos':True,'cache_prompt':False}
       pair['cold']=post(body)
       assert pair['cold']['timings']['prompt_n']==len(ids),('cold prompt reused',pair)
       pair['warm']=post(dict(body,cache_prompt=True))
       pair['vram_peak_mib']=peak.finish()
      except BaseException:peak.finish();raise
      pair['warm_prompt_cached']=pair['warm']['timings'].get('cache_n',0)>0
      if engine=='strata':pair['strata']=get('/props')['strata']
      result['requests'].append(pair);save(dst/'partial.json',result)
      print(f"{key} r{rep+1} {label}: pp={pair['cold']['timings']['prompt_per_second']:.1f} cold_tg={pair['cold']['timings']['predicted_per_second']:.1f} warm_tg={pair['warm']['timings']['predicted_per_second']:.1f} warm_read={pair['warm']['timings']['prompt_n']} peak={pair['vram_peak_mib']}",flush=True)
    save(completed,result);manifest['cells'][key]=result;save(a.out/'manifest.json',manifest);break
   except BaseException as exc:
    attempt['run_error']=str(exc);save(dst/'fit.json',attempts)
    # Preserve failures; only a proven VRAM failure permits the next static offload layout.
    tail=(logpath).read_text(errors='replace')[-12000:].lower()
    if engine=='llama' and any(s in tail for s in ('out of memory','failed to allocate','unable to allocate')):
     print(f'{key} ncm={ncm}: VRAM failure, retain log and increase CPU expert blocks',flush=True)
    else:raise
   finally:stop(proc,log)
  else:raise RuntimeError(f'{key}: no fitting placement')
manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(a.out/'manifest.json',manifest)
print('BENCHMARK COMPLETE',flush=True)
