#!/usr/bin/env bash
# TabPFN-3.5 continued pre-training, exactly as run on 2026-10-06/07
# (logs/cpt_w.log, logs/cpt_xw.log). One 96 GB GPU; tabpfn==9.0.0, torch 2.14.1.
# Inputs: data/ as described in README.md. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
REV=06bf2ba35c80a92a3b9abb436b99cf49e7a0365e
BASE=base/tabpfn-v3.5-20260909_multiclass-bf16.safetensors
mkdir -p base runs

# 0. The released multiclass checkpoint, cast to bfloat16 (header metadata kept).
#    Expected sha256 of $BASE: e543f27aa69074f69812dea4df03eb1cc715688010d8d26b7e7445dc2f7b6697
if [ ! -f "$BASE" ]; then
  $PY -c "from huggingface_hub import hf_hub_download as d; d('Prior-Labs/tabpfn_3_5', 'tabpfn-v3.5-20260909_multiclass.safetensors', revision='$REV', local_dir='base')"
  $PY recast_bf16.py base/tabpfn-v3.5-20260909_multiclass.safetensors "$BASE"
fi

# 1. 40,000 steps on 14 sources.
$PY cpt2.py --ckpt "$BASE" --out runs/cpt_w \
  --sources r1,r2,r3,r4,r5,r6,r7,r8,r10,hps34,hps40,nhis22,nhis23,piaac22 \
  --gated 0.3 --binned 0.35 --steps 40000 --save 0,10000,20000,30000,40000 --lr 3e-7 --seed 5 \
  > runs/cpt_w.log 2>&1

# 2. 20,000 more steps on 19 sources (steps are numbered 40,001-60,000).
$PY cpt2.py --ckpt runs/cpt_w/step40000.ckpt --out runs/cpt_xw --offset 40000 \
  --sources r1,r2,r3,r4,r5,r6,r7,r8,r10,hps12,hps34,hps40,nhis19,nhis21,nhis22,nhis23,piaac22,pisa18,pisa22 \
  --gated 0.3 --binned 0.35 --steps 20000 --save 10000,20000 --lr 3e-7 --seed 9 \
  > runs/cpt_xw.log 2>&1

# 3. bfloat16 copy = the archive file cpt_xw_step60000-bf16.ckpt.
$PY to_bf16.py runs/cpt_xw/step60000.ckpt runs/cpt_xw_step60000-bf16.ckpt
