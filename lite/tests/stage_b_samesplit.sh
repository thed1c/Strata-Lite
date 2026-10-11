#!/bin/bash
# Strata-Lite Gate 1 stage b, long prompts, the same execution grouping on both sides (TDUMP=- TDUMP_N=-2: both split
# their graphs at the same named nodes; past ~4.6k tokens llama.cpp's own output depends on that grouping). Full prompts -> llama.cpp greedy
# 200 tokens with logits -> strata-gemma4 teacher-forced and greedy.
export TDUMP=- TDUMP_N=-2
E=$WORK; cd $E || exit 1; O=$E/gate1b/samesplit; mkdir -p $O
M=$MODELS/gemma4-26b-a4b-qat-hauhaucs-balanced-mtp-f909366/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf
export CUDA_VISIBLE_DEVICES=GPU-<uuid>
NCM=${NCM:-30}
for id in ${IDS:-p3 p4 p5}; do
  P=$O/$id.ncm$NCM
  python3 -c "
import struct,sys
b=open('$E/gate1b/$id.ncm30.oracle.tokens','rb').read(); n,g=struct.unpack('<II',b[:8]); ids=struct.unpack('<%dI'%(n+g),b[8:])
L=n; open('$P.prompt.tokens','wb').write(struct.pack('<II',L,0)+struct.pack('<%dI'%L,*ids[:L]))"
  ROUTE_LOGITS=1 ROUTE_FORCE=$P.prompt.tokens ROUTE_OUT=$P.oracle ROUTE_NGEN=200 timeout 1800 numactl --cpunodebind=1 --membind=1 \
    llama.cpp/build/bin/route-oracle -m $M -ngl 99 --n-cpu-moe $NCM -fa on -c 8192 -ctk f16 -ctv f16 -b 2048 -ub 512 -t 16 --temp 0 > $P.oracle.log 2>&1
  for mode in force greedy; do
    timeout 1800 numactl --cpunodebind=1 --membind=1 build-g4/strata-gemma4 -m $M --tokens $P.oracle.tokens --oracle $P.oracle.logits \
      --mode $mode --n-cpu-moe $NCM -c 8192 -t 16 --out $P.g4-$mode > $P.g4-$mode.log 2>&1
    echo "$id(same-split) ncm=$NCM $mode: $(cat $P.g4-$mode.json 2>/dev/null || tail -2 $P.g4-$mode.log)"
  done
  rm -f $P.oracle.logits
done
