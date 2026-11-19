#!/bin/zsh
# E1/E2: per-track fine-tuning, 3 seeds, for each encoder. Logs -> results/logs/, log-probs -> results/runs/.
# Usage: zsh scripts/run_main.sh [model_key ...]   (default: all)
set -e
cd "${0:A:h}/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src
typeset -A MODELS LRS
MODELS=(xlmr FacebookAI/xlm-roberta-base  mdeberta microsoft/mdeberta-v3-base  mdeberta-xnli MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)
LRS=(xlmr ${XLMR_LR:-2e-5}  mdeberta ${MDEB_LR:-5e-5}  mdeberta-xnli ${XNLI_LR:-2e-5})
keys=(${@:-xlmr mdeberta mdeberta-xnli})
for key in $keys; do
  for seed in 1 2 3; do
    for track in track1 track2 track3 track4; do
      if [[ $track == track4 ]]; then epochs=${T4_EPOCHS:-4}; else epochs=${MONO_EPOCHS:-10}; fi
      run=${key}_${track}_s${seed}
      [[ -f results/runs/$run/train_log.json ]] && { echo "skip $run"; continue; }
      echo "$(date +%H:%M) $run"
      .venv/bin/python src/train.py --model ${MODELS[$key]} --train_tracks $track --run_name $run \
        --seed $seed --epochs $epochs --lr ${LRS[$key]} > results/logs/$run.log 2>&1
      tail -1 results/logs/$run.log
    done
  done
done
