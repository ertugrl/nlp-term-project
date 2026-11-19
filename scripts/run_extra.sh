#!/bin/zsh
# Runs queued after the main xlmr/mdeberta runs:
#   E2  mdeberta-xnli per track (LR picked from results/hp_track1.txt)
#   E4  mdeberta + symmetric-KL consistency loss (lambda=1) per track
#   E5  mdeberta trained on EN only, evaluated on all tracks (zero-shot transfer)
#   E5b mdeberta trained on all tracks pooled, evaluated on all tracks
cd "${0:A:h}/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src
XNLI_LR=$(python3 -c "
import ast
best=max((max(ast.literal_eval(l.split(' ',2)[2])), l.split()[1]) for l in open('results/hp_track1.txt') if l.startswith('mdeberta-xnli'))
print(best[1])")
echo "XNLI LR = $XNLI_LR"
XNLI_LR=$XNLI_LR zsh scripts/run_main.sh mdeberta-xnli
ALL=(track1 track2 track3 track4)
run() {  # run_name, extra args...
  local name=$1; shift
  [[ -f results/runs/$name/train_log.json ]] && { echo "skip $name"; return; }
  echo "$(date +%H:%M) $name"
  .venv/bin/python src/train.py --model microsoft/mdeberta-v3-base --lr 5e-5 --run_name $name "$@" > results/logs/$name.log 2>&1
  tail -1 results/logs/$name.log
}
for seed in 1 2 3; do
  for t in $ALL; do
    if [[ $t == track4 ]]; then ep=4; else ep=10; fi
    run mdeberta-cons_${t}_s${seed} --train_tracks $t --seed $seed --epochs $ep --cons_lambda 1.0
  done
  run mdeberta-en2all_s${seed} --train_tracks track1 --eval_tracks $ALL --seed $seed --epochs 10
  run mdeberta-pooled_s${seed} --train_tracks $ALL --seed $seed --epochs 4
done
echo "extra done"
