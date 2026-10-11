#!/bin/bash
cd $SERVING || exit 1
E=experiments/gemma4-5070; LOG=$E/sweep.log
say(){ echo "$(date +%T) [g5070] $*" | tee -a $LOG; }
for cfg in "20 32768" "10 32768" "20 262144"; do set -- $cfg; N=$1; CTX=$2
  tmux kill-session -t gemma-5070 2>/dev/null; sleep 3; pkill -f "port 18190" 2>/dev/null; sleep 2
  tmux new-session -d -s gemma-5070 -c $SERVING "bash $SERVING/$E/run-static.sh $N $CTX 2>&1 | tee $E/static-n$N-c$CTX.log; read"
  say "started n_cpu_moe=$N ctx=$CTX"; sleep 40
  python3 $E/bench.py $N $CTX $E/result-n$N-c$CTX.json 2>&1 | tail -2 | while read l; do say "$l"; done
done
say "G5070 SWEEP DONE"
