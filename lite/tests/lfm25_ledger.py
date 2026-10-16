#!/usr/bin/env python3
"""Recreate the public LFM ledger prompts used by lfm25_bench.py.

First run `python3 lite/tests/lfm25_ledger.py request ledger.json`.
Tokenize that JSON with route-oracle (ROUTE_TOKENIZE_ONLY=1), then run
`python3 lite/tests/lfm25_ledger.py split ledger.tokens OUTPUT_DIRECTORY`.
The selected prompts retain the last 16 tokens of the full request.
"""
import json,struct,sys
from pathlib import Path
mode=sys.argv[1]
if mode=='request':
 lines=[f'Archive record {i:05d}: The field team measured {17+i%29} units near station {i%97:02d}. They checked the instrument, recorded the reading, and stored the sample for later comparison.\n' for i in range(10000)]
 body={'messages':[{'role':'user','content':'Read the following synthetic research ledger.\n'+''.join(lines)+'\nSummarize the repeated procedure in two sentences.'}],'chat_template_kwargs':{'enable_thinking':True}}
 Path(sys.argv[2]).write_text(json.dumps(body))
elif mode=='split':
 b=Path(sys.argv[2]).read_bytes();n,g=struct.unpack_from('<II',b);ids=list(struct.unpack_from('<'+'I'*(n+g),b,8))
 out=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True)
 assert n>=120000
 for length in (5000,15000,65536,120000):
  chosen=ids[:length-16]+ids[-16:]
  (out/f'long-{length}.tokens').write_bytes(struct.pack('<II',length,0)+struct.pack('<'+'I'*length,*chosen))
else:raise SystemExit('mode must be request or split')
