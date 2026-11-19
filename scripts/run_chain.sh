#!/bin/zsh
# XNLI-init LR check on track1, then main runs for xlmr and mdeberta.
cd "${0:A:h}/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src
for lr in 2e-5 5e-5; do
  .venv/bin/python src/train.py --model MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 --train_tracks track1 \
    --run_name _hp_mdeberta-xnli_lr${lr} --epochs 10 --lr $lr > results/logs/_hp_mdeberta-xnli_lr${lr}.log 2>&1
  grep '"epoch"' results/logs/_hp_mdeberta-xnli_lr${lr}.log | python3 -c "import sys,json;print('mdeberta-xnli $lr', [round(json.loads(l)['val_mean_weighted_f1'],3) for l in sys.stdin])" | tee -a results/hp_track1.txt
done
zsh scripts/run_main.sh xlmr mdeberta
