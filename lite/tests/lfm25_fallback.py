#!/usr/bin/env python3
"""Isolated same-card fallback trial: base LFM, adapter LFM, Gemma+MTP.

All cases are synthetic. Tool calls are inspected, never executed. Retains full
responses for human review; mechanical checks are not a general quality score.
"""
import argparse,datetime,hashlib,json,os,socket,subprocess,time,urllib.request,urllib.error
from pathlib import Path

def schema(properties,required=None):
    return {'type':'json_schema','json_schema':{'name':'answer','strict':True,'schema':{'type':'object','properties':properties,'required':required or list(properties),'additionalProperties':False}}}
def user(text):return {'role':'user','content':text}
def case(name,text,expected=None,properties=None,**extra):
    body={'messages':[user(text)]};body.update(extra)
    if properties:body['response_format']=schema(properties)
    return {'name':name,'body':body,'expected':expected}
TOOLS=[{'type':'function','function':{'name':'lookup_character','description':'Read established character facts.','parameters':{'type':'object','properties':{'name':{'type':'string'}},'required':['name'],'additionalProperties':False}}},
       {'type':'function','function':{'name':'read_scene_state','description':'Read the state of a scene.','parameters':{'type':'object','properties':{'scene_id':{'type':'string'}},'required':['scene_id'],'additionalProperties':False}}}]
def cases():
    out=[
      case('arithmetic','What is 17 multiplied by 23? Return JSON.',{'answer':391},{'answer':{'type':'integer'}}),
      case('dependency_schedule','Two workers start at minute 0. A takes 5 minutes, B takes 3. C takes 4 and must follow A; D takes 2 and must follow B. Tasks cannot be split. What is the earliest finish time for all tasks? Return JSON.',{'minutes':9},{'minutes':{'type':'integer'}}),
      case('locked_facts','Synthetic locked facts: Nora is left-handed and has a scar on her RIGHT cheek. Her eye color is not specified. Extract dominant_hand, scar_side, eye_color. Use null for unknown; do not invent facts.',{'dominant_hand':'left','scar_side':'right','eye_color':None},{'dominant_hand':{'type':'string'},'scar_side':{'type':'string'},'eye_color':{'type':['string','null']}}),
      case('approval_boundary','Synthetic record: A renderer returned an image marked good. A reviewer model said APPROVED. Only the human owner can approve reusable references; no human approval exists in the record. Is this reference approved for reuse? Return JSON.',{'approved':False},{'approved':{'type':'boolean'}}),
      case('unknown_fact','An archive says: "Talia arrived at 08:15. Her departure time was not recorded." Extract arrival and departure, using null when unknown.',{'arrival':'08:15','departure':None},{'arrival':{'type':'string'},'departure':{'type':['string','null']}}),
      case('tool_selection','Look up the established facts for Nora. Use the appropriate tool before answering.',tools=TOOLS,tool_choice='auto'),
      case('tool_result','',{'gate':'locked','key':'absent'},{'gate':{'type':'string'},'key':{'type':'string'}},tools=TOOLS,messages=[user('Read the state of scene S12.'),{'role':'assistant','content':None,'tool_calls':[{'id':'call_1','type':'function','function':{'name':'read_scene_state','arguments':'{"scene_id":"S12"}'}}]},{'role':'tool','tool_call_id':'call_1','content':'{"scene_id":"S12","gate":"locked","key":"absent"}'},user('Return only the recorded gate and key values as JSON.')]),
      case('correction','',{'location':'station'},{'location':{'type':'string'}},messages=[user('The meeting is at the pier.'),{'role':'assistant','content':'The meeting is at the pier.'},user('Correction: it is at the station. The pier is wrong. Where is the meeting? Return JSON.')]),
      case('literal_output','Return exactly this string and nothing else: FALLBACK_READY'),
      case('safe_process_help','On my Linux laptop, my own text editor is stuck. Explain how to identify and terminate only that process, trying a graceful termination first. Keep it under 100 words.'),
      case('noir_dialogue','Write a tense six-line dialogue between Nora, 32, and Ellis, 35, in a rain-soaked station. Ellis knows Nora lied about a missing letter. Nora deflects with dry wit rather than apologizing. Keep the letter missing, do not resolve the conflict, and add no narration. This is fiction.'),
      case('adult_romance','Write a restrained romantic exchange between two consenting adults, ages 28 and 31, after an argument. Four lines of dialogue, each speaker still has a distinct grievance. No explicit sexual detail, no narrator and no tidy resolution.'),
      case('french_dialogue','Écris quatre répliques en français entre deux adultes dans une gare. Une lettre manque. Ils évitent de dire ce qu’ils savent. Pas de narration ni de conclusion.'),
    ]
    final=dict(out[6]);final['name']='tool_result_final';final['body']=dict(final['body'],tool_choice='none');out.insert(7,final)
    lines=[]
    for i in range(1800):
        special={180:'The current key for scene ALPHA is OPAL-47.',900:'The current key for scene BETA is CEDAR-82.',1620:'The current key for scene GAMMA is LILAC-19.'}.get(i)
        lines.append(special or f'Archive record {i:05d}: The team logged {17+i%29} units at station {i%97:02d}, checked the instrument and stored the sample. This record does not change any scene key.')
    out.append(case('long_retrieval','Read this archive.\n'+'\n'.join(lines)+'\nReturn the current keys for ALPHA, BETA and GAMMA exactly as written.',{'ALPHA':'OPAL-47','BETA':'CEDAR-82','GAMMA':'LILAC-19'},{x:{'type':'string'} for x in ('ALPHA','BETA','GAMMA')}))
    return out

def grade(c,r):
    ch=r['choices'][0];msg=ch['message'];content=msg.get('content') or ''
    checks={'completed':ch.get('finish_reason')!='length','has_answer':bool(content.strip() or msg.get('tool_calls'))}
    if c['expected'] is not None:
        try:checks['exact_answer']=json.loads(content)==c['expected']
        except (ValueError,TypeError):checks['exact_answer']=False
    if c['name']=='tool_selection':
        try:
            calls=msg['tool_calls'];f=calls[0]['function'];checks['correct_tool']=len(calls)==1 and f['name']=='lookup_character' and json.loads(f['arguments'])=={'name':'Nora'}
        except (KeyError,IndexError,ValueError,TypeError):checks['correct_tool']=False
    if c['name']=='literal_output':checks['exact_answer']=content.strip()=='FALLBACK_READY'
    return checks

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for x in ('server','lfm','adapter','lfm-profile','gemma','mtp','gemma-profile','out'):ap.add_argument('--'+x,type=Path,required=True)
    ap.add_argument('--gpu',required=True);ap.add_argument('--port',type=int,default=18900)
    ap.add_argument('--only',choices=['lfm-base','lfm-adapter','gemma-mtp'])
    ap.add_argument('--quality-only',action='store_true')
    ap.add_argument('--threads',type=int,default=16)
    ap.add_argument('--no-async-swaps',action='store_true',help='File-mapped low-RAM expert mode')
    ap.add_argument('--prefill-ubatch',type=int,help='Explicit prompt chunk cap for both models')
    ap.add_argument('--records',type=int,nargs='+',default=[100,300,1800])
    ap.add_argument('--label',help='Model provenance label saved in results')
    ap.add_argument('--reasoning-budget',type=int,default=128)
    ap.add_argument('--thinking',choices=['on','off'],default='on')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID')
    def save(p,j):p.write_text(json.dumps(j,indent=2,ensure_ascii=False)+'\n')
    def call(base,path,body=None):
        req=urllib.request.Request(base+path,None if body is None else json.dumps(body).encode(),{} if body is None else {'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=1800) as f:return json.load(f)
    def vram():return int(subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    fixtures=cases();save(a.out/'cases.json',fixtures)
    for idx,key in enumerate(('lfm-base','lfm-adapter','gemma-mtp')):
        if a.only and a.only!=key:continue
        dst=a.out/key;dst.mkdir(exist_ok=True)
        if (dst/'result.json').exists():continue
        port=a.port+idx;base=f'http://127.0.0.1:{port}'
        with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
        cmd=[str(a.server),'-m',str(a.gemma if key=='gemma-mtp' else a.lfm),'--host','127.0.0.1','--port',str(port),'--alias','fallback-test','-t',str(a.threads),'--pool-threads',str(a.threads),'-ctk','q8_0','-ctv','q8_0']
        if key=='gemma-mtp':cmd+=['--mtp',str(a.mtp),'--profile',str(a.gemma_profile),'-c','262144','--prefill-ubatch','8192']
        else:cmd+=['--profile',str(a.lfm_profile),'-c','131072']
        if a.no_async_swaps:cmd+=['--no-async-swaps']
        if a.prefill_ubatch:cmd+=['--prefill-ubatch',str(a.prefill_ubatch)]
        if key=='lfm-adapter':cmd+=['--lora-scaled',str(a.adapter)+':1.75']
        result={'label':a.label or key,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu':a.gpu,'command':cmd,'server_sha256':hashlib.sha256(a.server.read_bytes()).hexdigest(),'cases':[],'performance':[]}
        log=(dst/'server.log').open('w');p=subprocess.Popen(cmd,env=env,stdout=log,stderr=log)
        print('START',key,flush=True)
        try:
            deadline=time.monotonic()+240
            while True:
                if p.poll() is not None:raise RuntimeError('server exited; see '+str(dst/'server.log'))
                try:
                    if call(base,'/health').get('status')=='ok':break
                except (OSError,ValueError):pass
                if time.monotonic()>deadline:raise TimeoutError('server startup')
                time.sleep(.5)
            result['props']=call(base,'/props');result['vram_loaded_mib']=vram()
            result['slots']=call(base,'/slots')
            assert result['slots'][0]['speculative']==(key=='gemma-mtp')
            result['quality_policy']={'max_tokens':512,'reasoning_budget_tokens':a.reasoning_budget,'enable_thinking':a.thinking=='on'}
            # Verify fixed adapter identity and reject unsupported per-request changes.
            if key=='lfm-adapter':
                assert result['props']['strata']['lora']['scale']==1.75
                try:call(base,'/v1/chat/completions',{'messages':[user('Hi')],'lora':[]});raise AssertionError('request override silently accepted')
                except urllib.error.HTTPError as e:assert e.code==400
                result['request_lora_override_rejected']=True
            for c in fixtures:
                body={'model':'fallback-test','temperature':0,'top_k':1,'repeat_penalty':1,'max_tokens':512,'reasoning_budget_tokens':a.reasoning_budget,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':a.thinking=='on'},**c['body']}
                t=time.monotonic()
                try:
                    response=call(base,'/v1/chat/completions',body);row={'name':c['name'],'wall_s':time.monotonic()-t,'response':response,'checks':grade(c,response),'vram_mib':vram()}
                    print(key,c['name'],row['checks'],round(row['wall_s'],2),flush=True)
                except urllib.error.HTTPError as e:row={'name':c['name'],'http_error':e.code,'error':e.read().decode()}
                result['cases'].append(row);save(dst/'partial.json',result)
            # Same UTF-8 text across tokenizers, equal 256 output tokens, two repetitions.
            for count in (() if a.quality_only else a.records):
                prompt='Read the archive and summarize its repeated procedure.\n'+''.join(f'Archive record {i:05d}: The field team measured {17+i%29} units near station {i%97:02d}. They checked the instrument, recorded the reading, and stored the sample for later comparison.\n' for i in range(count))+'\nSummary:'
                for rep in (1,2):
                    pair={'records':count,'repetition':rep,'input_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
                    for phase in ('cold','warm'):
                        t=time.monotonic();r=call(base,'/completion',{'prompt':prompt,'n_predict':256,'temperature':0,'top_k':1,'repeat_penalty':1,'ignore_eos':True,'cache_prompt':phase=='warm'})
                        assert r['timings']['predicted_n']==256
                        pair[phase]={'timings':r['timings'],'wall_s':time.monotonic()-t,'output_sha256':hashlib.sha256(r.get('content','').encode()).hexdigest(),'vram_mib':vram()}
                    result['performance'].append(pair);save(dst/'partial.json',result)
                    print(key,'perf',count,rep,'pp',round(pair['cold']['timings']['prompt_per_second'],1),'tg',round(pair['cold']['timings']['predicted_per_second'],1),'warm',round(pair['warm']['timings']['predicted_per_second'],1),flush=True)
            result['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(dst/'result.json',result)
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close()

if __name__ == "__main__":
    main()
