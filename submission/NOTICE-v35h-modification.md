# TabPFN-3.5 (Prior Labs GmbH)

The TABPFN-3.5 Model is licensed by Prior Labs GmbH under the TABPFN-3.5
Non-Commercial License.
Copyright (c) Prior Labs GmbH 2026.

THE SERVICES ARE PROVIDED FREE OF CHARGE: COMPANY SHALL NOT BE LIABLE FOR
DAMAGES RESULTING FROM SLIGHT NEGLIGENCE. LIABILITY FOR GROSS NEGLIGENCE AND
INTENTIONAL MISCONDUCT REMAINS UNAFFECTED.

The full licence is distributed alongside these weights as `LICENSE-v35.txt`.
Any rights to use the TABPFN-3.5 Model are granted directly by Prior Labs GmbH
to the recipient under that licence, not by us.

Source: https://huggingface.co/Prior-Labs/tabpfn_3_5 (revision
06bf2ba35c80a92a3b9abb436b99cf49e7a0365e, published 2026-09-09).

## Modification statement (License section 3.c.i)

If the archive carries `tabpfn-v3.5-20260909-bf16.safetensors` or
`tabpfn-v3.5-20260909_multiclass-bf16.safetensors`, that file is a MODIFIED copy
of `tabpfn-v3.5-20260909.safetensors` / `tabpfn-v3.5-20260909_multiclass.safetensors`
respectively: every floating-point
tensor was cast from float32 to bfloat16 and re-serialised, with the original
safetensors header metadata preserved. No weight values were retrained,
pruned or otherwise altered. The recast exists only to fit the 1 GB submission
archive; TabPFN already runs inference under bfloat16 autocast.

This is not an official product of Prior Labs GmbH, and it has not been
endorsed, approved or validated by Prior Labs GmbH.
