#!/usr/bin/env bash
# Kumo-Tabular (large classifier) continued pre-training, exactly as run on
# 2026-10-08/09 (logs/kcpt_a.log, kcpt_a2.log, kcpt_kd.log). One 96 GB GPU;
# sdm (structured-data-models) = submission/sdm, torch 2.14.1.
# The start point is nvidia/Kumo-Tabular v1.0.1 (b2f5a9d6...), which sdm
# downloads when --init is not given. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd .. && pwd)/submission${PYTHONPATH:+:$PYTHONPATH}"   # for sdm
PY=${PY:-python}
SRC=r1,r2,r3,r4,r5,r6,r7,r8,r10,hps12,hps34,hps40,nhis19,nhis21,nhis22,nhis23,piaac22,pisa18,pisa22,wgm18
mkdir -p runs

# 1. 15,000 steps from the released weights.
$PY cpt_kumo.py --out runs/kcpt_a --sources $SRC --gated 0.3 --binned 0.35 \
  --steps 15000 --save 2500,5000,10000,15000 --lr 1e-6 --seed 11 > runs/kcpt_a.log 2>&1

# 2. 15,000 more steps (30,000 in total).
$PY cpt_kumo.py --out runs/kcpt_a2 --init runs/kcpt_a/step15000.safetensors --sources $SRC \
  --gated 0.3 --binned 0.35 --steps 15000 --save 2500,5000,10000,15000 --lr 1e-6 --seed 21 \
  > runs/kcpt_a2.log 2>&1

# 3. Larger tasks (4,000-15,000 rows). The run went to 10,000 steps; the archive
#    uses the step-5,000 save (35,000 in total). The schedule is warm-up then
#    constant, so the step-5,000 save does not depend on the run length.
$PY cpt_kumo.py --out runs/kcpt_kd --init runs/kcpt_a2/step15000.safetensors \
  --max_rows 15000 --rows 4000,15001 --sources $SRC --gated 0.3 --binned 0.35 \
  --steps 10000 --save 5000,10000 --lr 1e-6 --seed 31 > runs/kcpt_kd.log 2>&1

# cpt_kumo.py saves bfloat16 safetensors, so this file is the archive file
# kumo-tabular-large-classifier-bf16.safetensors (sha256 f72465b4...).
sha256sum runs/kcpt_kd/step05000.safetensors
