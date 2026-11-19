# TabPFN-3.5 continued-pretrained checkpoint (cpt_xw_step60000-bf16.ckpt)

The file `cpt_xw_step60000-bf16.ckpt` in this archive is a MODIFIED copy of the
TABPFN-3.5 Model by Prior Labs GmbH, licensed under the TABPFN-3.5
Non-Commercial License (full text: `LICENSE-v35mh.txt`; see also
`NOTICE-v35mh.txt` and `NOTICE-v35h-modification.md`).
Copyright (c) Prior Labs GmbH 2026.

Source: https://huggingface.co/Prior-Labs/tabpfn_3_5 (revision
06bf2ba35c80a92a3b9abb436b99cf49e7a0365e, published 2026-09-09), file
`tabpfn-v3.5-20260909_multiclass.safetensors`.

## Modification statement (License section 3.c.i)

The weights were changed by continued pre-training (all parameters, AdamW,
learning rate 3e-7, one task per step) and are therefore not the released
values:

1. 40,000 steps from the released multiclass checkpoint (cast to bfloat16) on
   public survey tables: Afrobarometer rounds 1-8 and 10, US Census Household
   Pulse Survey (weeks 34, 40), NHIS 2022 and 2023, PIAAC Cycle 2 (2022).
2. 20,000 further steps on the same sources plus Household Pulse week 12,
   NHIS 2019 and 2021, PISA 2018 and 2022.

The result was saved as bfloat16. Architecture, preprocessing and inference
code are unchanged (`tabpfn==9.0.0`). No competition data were used in this
training. The archive maps both of its TabPFN-3.5 backbones (`v35h`, `v35mh`
in `main.py`) to this one checkpoint.

This is not an official product of Prior Labs GmbH, and it has not been
endorsed, approved or validated by Prior Labs GmbH.
