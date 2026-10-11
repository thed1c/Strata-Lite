#!/bin/bash
# Strata-Lite public grid for Qwen3.6-35B-A3B (HauhauCS Aggressive Q4_K_P + unsloth's MTP head extracted; the maintainer,
# 2026-10-10: "Go for the Qwen grid"), tests/grid.sh's method and cells unchanged; Qwen's 40 expert blocks, its
# Qwen-tokenized prompts (q35g/) and its all-GPU q4 references (q35kv/, g157q/).
# Gemma's header follows.
# Strata-Lite public grid (the maintainer, 2026-10-10: "Apples to Apples on the 12Gb cards").
# One 12 GB card, contexts 32k / 131k / 262k (q4 KV everywhere: the references are q4), two engines with MTP on:
#   lite   Strata-Lite (this port): the most expert slots the card holds (auto), chunked reads with lent VRAM at the
#          largest chunk the card funds (16k, 8k, 4k tried in that order)
#   llama  llama.cpp: static expert offload at the most it can hold (the fewest --n-cpu-moe blocks that load and run),
#          -b 2048 -ub 512 (its defaults), draft-mtp
# Per cell: speed (grid_bench.py: the 5k, 15k and, at 262k, the 157k prompt as token ids, cold then warm, 256 greedy
# tokens; VRAM at load and peak), quality (teacher-forced through llama.cpp all-GPU q4's greedy tokens: top-1 and KL on
# the five prompts and, at 262k, the 157k prompt; Strata-Lite with the twin check on), and the image projector's fit at
# that context (the same config + the mmproj on the card: does it load and answer one and eight 1024 px images).
# usage: CARD=<uuid> TAG=5070 bash grid_qwen.sh   (env: CTXS, ENGINES, NUMA, LLAMA, G4, Q35, STEPS="speed quality vision")
set -u
E=$WORK; cd $E || exit 1
TAG=${TAG:?}; CARD=${CARD:?}; O=$E/grid-qwen/$TAG; mkdir -p $O; LOG=$O/run.log
exec > >(tee -a "$LOG") 2>&1
exec 3>&1
MD=$MODELS/qwen36-35b-a3b-hauhaucs-aggressive
M=$MD/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf; MTP=$E/qwen36/mtp-qwen3.6-35b-a3b-unsloth.gguf
MMPROJ=$MD/mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf; PROFILE=$E/packs/qwen36/profile-canon.txt; NB=40
LLAMA=${LLAMA:-$E/llama.cpp/build86/bin}; G4=${G4:-$E/build-g4-86}; Q35=${Q35:-$E/build-q35}; NUMA=${NUMA-numactl --cpunodebind=1 --membind=1}
CTXS=${CTXS:-32768 131072 262144}; ENGINES=${ENGINES:-lite llama}; STEPS=${STEPS:-speed quality vision}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD
FIVE="p1 p2 p3 p4 p5"
PORT=18195
say(){ echo "$(date +%T) [grid $TAG] $*"; }
PIDS=()
cleanup(){ # stop our servers and wait until they are gone and the port is free (freeing 15 GB of pinned memory takes seconds,
  # and the dying process keeps its socket until then)
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done; sleep 3; for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null; done
  local t=0; while [ $t -lt 120 ]; do local busy=0
    for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && busy=1; done
    ss -ltn | /bin/grep -q "127.0.0.1:$PORT " && busy=1
    [ $busy = 0 ] && break; sleep 2; t=$((t+2)); done
  PIDS=(); sleep 2; }
trap 'say "STOPPED: interrupted"; cleanup; echo "STOPPED: interrupted"; exit 1' INT TERM
waitrun(){ local max=$1 what=$2; shift 2; "$@" & local p=$! t=0
  while kill -0 $p 2>/dev/null; do sleep 10; t=$((t+10)); [ $((t % 120)) = 0 ] && say "  ... $what ${t}s" >&3; [ $t -ge $max ] && { kill $p; sleep 2; kill -9 $p 2>/dev/null; say "  $what: deadline ${max}s" >&3; return 124; }; done
  wait $p; }
start(){ # logfile cmd... -> 0 when /health answers, 1 when the process died or did not come up in 10 min; a port that is
  # not free yet (a dying server) is retried, never read as a fit result
  local try; for try in 1 2 3; do start1 "$@" && return 0
    /bin/grep -q "couldn't bind\|cannot listen" $1 || return 1
    say "  port $PORT not free yet; again in 20 s"; cleanup; sleep 20; done; return 1; }
start1(){
  local lf=$1; shift; $NUMA "$@" > $lf 2>&1 & local p=$!; PIDS+=($p); local t=0
  while :; do curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null | /bin/grep -q 200 && return 0
    kill -0 $p 2>/dev/null || return 1
    sleep 3; t=$((t+3)); [ $t -ge 600 ] && return 1; done; }
alive(){ curl -s -o /dev/null http://127.0.0.1:$PORT/health; }
ok_all(){ python3 -c "import json,sys; d=json.load(open('$1')); r=d['requests']; sys.exit(0 if r and all(v['cold'].get('code')==200 for v in r.values()) else 1)" 2>/dev/null; }
prompts(){ local c=$1 p="p5k=$E/q35g/p5k.tokens,p15k=$E/q35g/p15k.tokens"; [ $c -ge 160000 ] && p="$p,p157k=$E/q35g/long-a.tokens"; echo $p; }
lite_cmd(){ # ctx chunk [mmproj]
  local c=$1 P=$2 mm=${3:-}; echo $G4/strata-lite-server -m $M --mtp $MTP ${mm:+--mmproj $MMPROJ --image-max-tokens 280} --alias grid --host 127.0.0.1 --port $PORT \
    -c $c --profile $PROFILE --cache-slots auto -t 16 --pool-threads 16 --prefill-ubatch $P; }
llama_cmd(){ # ctx N [mmproj]
  local c=$1 N=$2 mm=${3:-}; echo $LLAMA/llama-server -m $M -md $MTP --spec-type draft-mtp ${mm:+--mmproj $MMPROJ --image-max-tokens 280} --alias grid \
    --host 127.0.0.1 --port $PORT -ngl 99 -fa on -c $c --n-cpu-moe $N -t 16 --cache-type-k q4_0 --cache-type-v q4_0 --parallel 1 -b 2048 -ub 512 \
    --fit off --no-context-shift --jinja --no-webui --timeout 7200; }
nstart(){ case $1 in 32768) echo ${N32:-24};; 131072) echo ${N131:-25};; *) echo ${N262:-26};; esac; }

say "grid on $TAG ($CARD): contexts $CTXS, engines $ENGINES, steps $STEPS"
ss -ltn | /bin/grep -q "127.0.0.1:$PORT " && { say "STOPPED: port $PORT is taken"; echo "STOPPED: port $PORT is taken"; exit 1; }
for c in $CTXS; do
  for eng in $ENGINES; do
    D=$O/c$c/$eng; mkdir -p $D
    # ---------------------------------------------------------------- speed: the most the card holds
    if [[ " $STEPS " == *" speed "* ]] && [ ! -s $D/speed.json ]; then
      if [ $eng = lite ]; then
        for P in 16384 8192 4096; do
          say "$c lite: chunk $P, auto slots"
          if start $D/server-P$P.log $(lite_cmd $c $P); then
            waitrun 3600 "$c lite bench" python3 $E/tests/grid_bench.py http://127.0.0.1:$PORT $D/speed-P$P.json --card $CARD --prompts $(prompts $c) --label lite-c$c-P$P
            if ok_all $D/speed-P$P.json && alive; then cp $D/speed-P$P.json $D/speed.json; echo $P > $D/chunk; cleanup; break; fi
            say "  chunk $P did not hold: $(/bin/grep -m1 -i 'out of memory\|error\|assert' $D/server-P$P.log | cut -c1-160)"
          else say "  chunk $P: server did not start: $(tail -1 $D/server-P$P.log | cut -c1-160)"; fi
          cleanup
        done
      else
        N=$(nstart $c)
        while [ $N -le $NB ]; do
          say "$c llama: $N of $NB expert blocks in RAM"
          if start $D/server-N$N.log $(llama_cmd $c $N); then
            waitrun 5400 "$c llama bench" python3 $E/tests/grid_bench.py http://127.0.0.1:$PORT $D/speed-N$N.json --card $CARD --prompts $(prompts $c) --label llama-c$c-N$N
            if ok_all $D/speed-N$N.json && alive; then cp $D/speed-N$N.json $D/speed.json; echo $N > $D/ncpumoe; cleanup; break; fi
            say "  N=$N did not hold"
          else say "  N=$N: did not load"; fi
          cleanup; N=$((N+1))
        done
      fi
      [ -s $D/speed.json ] && say "  $c $eng speed: $(python3 -c "
import json;d=json.load(open('$D/speed.json'));r=d['requests']
print('load %d MiB peak %d MiB;'%(d['vram_load_mib'],d['vram_peak_mib']), '; '.join('%s %.0f tok/s read, %.1f/%.1f decode'%(k,v['cold'].get('prompt_per_second',0),v['cold'].get('predicted_per_second',0),(v.get('warm') or {}).get('predicted_per_second',0)) for k,v in r.items()), d.get('props_strata') or '')")" || say "  $c $eng: no configuration held"
    fi
    # ---------------------------------------------------------------- quality: against llama.cpp all-GPU q4
    if [[ " $STEPS " == *" quality "* ]] && [ -s $D/speed.json ]; then
      ids="$FIVE"; [ $c -ge 160000 ] && ids="$ids g157"
      for id in $ids; do
        [ -s $D/q-$id.json ] && continue
        if [ $id = g157 ]; then T=$E/g157q/allgpu-q4.tokens; R=$E/g157q/allgpu-q4.logits; else T=$E/q35b/$id.oracle.tokens; R=$E/q35kv/$id.allgpu-q4.logits; fi
        if [ $eng = lite ]; then
          S=$(python3 -c "import json;print(json.load(open('$D/speed.json'))['props_strata']['slots'])"); P=$(cat $D/chunk)
          for s in $((S-8)) $((S-16)) $((S-24)); do   # the twin check's layer copy takes ~0.6 GB of the card
            waitrun 3600 "$c lite quality $id" $NUMA $Q35/strata-qwen35 -m $M --pool --pool-threads 16 --n-cpu-moe 40 -c $c -t 16 -ctk q4_0 -ctv q4_0 \
              --cache-slots $s --profile $PROFILE --adapt --async-swaps --overlap --prefill-ubatch $P --lend --mtp $MTP --n-draft 3 --mtp-share-head \
              --dcheck-layer 3 --tokens $T --oracle $R --mode mtp-force --out $D/q-$id > $D/q-$id.log 2>&1
            [ -s $D/q-$id.json ] && break
          done
          say "  $c lite $id: $(python3 -c "import json;d=json.load(open('$D/q-$id.json'));print('slots',d['cache_slots'],'top1 %d/%d'%(d['argmax_same'],d['steps_compared']),'KL %.2e'%d['kl_mean'],'twins',d['dcheck_rows'],'differ',d['dcheck_bad'],'hits %.3f'%d['hit_rate'])" 2>/dev/null || tail -1 $D/q-$id.log)"
        else
          N=$(cat $D/ncpumoe)
          ROUTE_LOGITS=1 ROUTE_FORCE=$T ROUTE_OUT=$D/q-$id waitrun 5400 "$c llama quality $id" $NUMA $LLAMA/route-oracle -m $M -ngl 99 --n-cpu-moe $N -fa on -c $c \
            -ctk q4_0 -ctv q4_0 -b 2048 -ub 512 -t 16 --temp 0 > $D/q-$id.log 2>&1
          $G4/logit-cmp $R $D/q-$id.logits 248320 > $D/q-$id.cmp 2>&1; rm -f $D/q-$id.logits $D/q-$id.routes
          tail -1 $D/q-$id.cmp | /bin/grep -q '^{' && tail -1 $D/q-$id.cmp > $D/q-$id.json
          say "  $c llama $id: $(tail -1 $D/q-$id.cmp)"
        fi
      done
    fi
    # ---------------------------------------------------------------- vision: the projector on the same card
    if [[ " $STEPS " == *" vision "* ]] && [ -s $D/speed.json ] && [ ! -s $D/vision.json ]; then
      if [ $eng = lite ]; then
        P=$(cat $D/chunk); say "$c lite + projector (chunk $P, auto slots)"
        if start $D/vision.log $(lite_cmd $c $P mm); then
          waitrun 1800 "$c lite vision" python3 $E/tests/grid_bench.py http://127.0.0.1:$PORT $D/vision.json --card $CARD --prompts p5k=$E/q35g/p5k.tokens --images --label lite-c$c-vision
        else echo '{"fits": false}' > $D/vision.json; fi
      else
        N=$(cat $D/ncpumoe); fit=0
        while [ $N -le $NB ]; do
          say "$c llama + projector, $N blocks in RAM"
          if start $D/vision-N$N.log $(llama_cmd $c $N mm); then
            waitrun 1800 "$c llama vision" python3 $E/tests/grid_bench.py http://127.0.0.1:$PORT $D/vision.json --card $CARD --prompts p5k=$E/q35g/p5k.tokens --images --label llama-c$c-N$N-vision
            if ok_all $D/vision.json && alive; then echo $N > $D/vision_ncpumoe; fit=1; cleanup; break; fi
          fi
          cleanup; N=$((N+1))
        done
        [ $fit = 1 ] || echo '{"fits": false}' > $D/vision.json
      fi
      say "  $c $eng projector: $(python3 -c "
import json;d=json.load(open('$D/vision.json'));r=d.get('requests',{})
print('fits' if r and all(v['cold'].get('code')==200 for v in r.values()) else 'does not fit', d.get('props_strata_end') or '', {k:v['cold'].get('wall_s') for k,v in r.items()})" 2>/dev/null)"
      cleanup
    fi
  done
done
cleanup
GRID_BLOCKS=$NB python3 $E/tests/grid_summary.py $E/grid-qwen > $E/grid-qwen/summary.md 2>&1 && say "summary: $E/grid-qwen/summary.md" || say "summary failed"
say "DONE: grid $TAG"
echo "DONE: grid $TAG"
