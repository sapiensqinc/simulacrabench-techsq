#!/usr/bin/env bash
# Download the three weight files that the archive ships, into weights/,
# and check each against MANIFEST.sha256.
#
#   bash scripts/fetch_weights.sh            # the three archive weights
#   bash scripts/fetch_weights.sh --v2-clf   # also TabPFN-v2-clf into the HF cache
#                                            # (needed only for a local run)
#
# Our two continued-pretrained checkpoints come from this repository's
# release. EXAONE-Tabular is unmodified, so it comes from Hugging Face at
# the pinned revision; we do not redistribute it.
set -euo pipefail
cd "$(dirname "$0")/.."
REL=${SB_RELEASE_URL:-https://github.com/sapiensqinc/simulacrabench-techsq/releases/download/test-1055650}
EXA=https://huggingface.co/LG-AI-Research/EXAONE-Tabular/resolve/093ad1c2613e1e95d26df8610451d2e02ed53cbe/exaone-tabular-classifier-v1_default.safetensors
mkdir -p weights

get() {  # get URL FILE
  if [ -f "weights/$2" ]; then echo "have weights/$2"; return; fi
  echo "fetch $2"
  curl -fL --retry 3 -o "weights/$2.part" "$1" && mv "weights/$2.part" "weights/$2"
}

get "$REL/cpt_xw_step60000-bf16.ckpt" cpt_xw_step60000-bf16.ckpt
get "$REL/kumo-tabular-large-classifier-bf16.safetensors" kumo-tabular-large-classifier-bf16.safetensors
get "$EXA" exaone-tabular-classifier-v1_default.safetensors

(cd weights && grep -E "  (cpt_xw_step60000-bf16.ckpt|kumo-tabular-large-classifier-bf16.safetensors|exaone-tabular-classifier-v1_default.safetensors)$" \
   ../MANIFEST.sha256 | sha256sum -c -)

if [ "${1:-}" = "--v2-clf" ]; then
  # models.txt names Prior-Labs/TabPFN-v2-clf; the hosted image prefetches it.
  python -c "from huggingface_hub import snapshot_download as s; print(s('Prior-Labs/TabPFN-v2-clf', revision='f851f2a3c941544733b712d8c0f96dfae9b28862'))"
fi
