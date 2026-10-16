import argparse,datetime,hashlib,json,pathlib,time,urllib.request
from lfm25_fallback import cases,grade
ap=argparse.ArgumentParser();ap.add_argument('--url',required=True);ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--label',required=True);ap.add_argument('--case',action='append');ap.add_argument('--quality',action='store_true');ap.add_argument('--performance',action='store_true');ap.add_argument('--reasoning-budget',type=int,default=128);ap.add_argument('--records',type=int,nargs='+',default=[100,300,600]);a=ap.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
def call(path,body=None):
 req=urllib.request.Request(a.url+path,None if body is None else json.dumps(body).encode(),{} if body is None else {'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=2400) as r:return json.load(r)
r={'label':a.label,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'props':call('/props'),'slots':call('/slots'),'cases':[],'performance':[]}
def save():a.out.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
if a.quality or a.case:
 for c in cases():
  if a.case and c['name'] not in a.case:continue
  body={'model':'fallback','temperature':0,'top_k':1,'repeat_penalty':1,'max_tokens':512,'reasoning_budget_tokens':a.reasoning_budget,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':a.reasoning_budget>0},**c['body']}
  t=time.monotonic();ans=call('/v1/chat/completions',body);row={'name':c['name'],'wall_s':time.monotonic()-t,'response':ans,'checks':grade(c,ans)};r['cases'].append(row);save();print(c['name'],row['checks'],round(row['wall_s'],2),flush=True)
if a.performance:
 for count in a.records:
  prompt='Read the archive and summarize its repeated procedure.\n'+''.join(f'Archive record {i:05d}: The field team measured {17+i%29} units near station {i%97:02d}. They checked the instrument, recorded the reading, and stored the sample for later comparison.\n' for i in range(count))+'\nSummary:'
  for rep in [1,2]:
   pair={'records':count,'repetition':rep,'input_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
   for phase in ['cold','warm']:
    t=time.monotonic();ans=call('/completion',{'prompt':prompt,'n_predict':256,'temperature':0,'top_k':1,'repeat_penalty':1,'ignore_eos':True,'cache_prompt':phase=='warm'});assert ans['timings']['predicted_n']==256
    pair[phase]={'timings':ans['timings'],'wall_s':time.monotonic()-t,'output_sha256':hashlib.sha256(ans.get('content','').encode()).hexdigest()}
   r['performance'].append(pair);save();print(count,rep,'pp',pair['cold']['timings']['prompt_per_second'],'tg',pair['cold']['timings']['predicted_per_second'],'warm',pair['warm']['timings']['predicted_per_second'],flush=True)
r['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print('COMPLETE',flush=True)
