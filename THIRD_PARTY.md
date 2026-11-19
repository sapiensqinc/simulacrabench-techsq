# Third-party code and weights

Our own code is under Apache-2.0 (`LICENSE`). It covers `submission/main.py`, `submission/stack.py`, `submission/cpu_worker.py`, `scripts/` and the scripts in `cpt/`. The components below keep their own licences. The licence texts ship with the archive and are in `submission/`.

## Code vendored in the archive

| Path | What | Version | Licence | Licence file |
|---|---|---|---|---|
| `submission/sdm/` | NVIDIA structured-data-models (Kumo-Tabular inference code), unmodified | `842c408fe2a8711bdf2e7cff4bfbe54266d6b940` | Apache-2.0 | `LICENSE-sdm-code.txt` |
| `submission/exaonetabular/` | EXAONE-Tabular inference code (LG AI Research) | as shipped with the model | BSD-3-Clause-LG AI Research License | `LICENSE-exa-code.txt`, `NOTICE-exa-thirdparty.md` |

## Weights

| Archive file | Model | Revision | Licence | Where to get it | Notes |
|---|---|---|---|---|---|
| `cpt_xw_step60000-bf16.ckpt` | TabPFN-3.5 multiclass, **modified by us** (CPT) | base `Prior-Labs/tabpfn_3_5` @ `06bf2ba35c80a92a3b9abb436b99cf49e7a0365e` | TABPFN-3.5 Non-Commercial License | this repository's release | `NOTICE-v35-cpt.md`, `LICENSE-v35mh.txt` |
| `kumo-tabular-large-classifier-bf16.safetensors` | Kumo-Tabular large classifier, **modified by us** (CPT) | base `nvidia/Kumo-Tabular` v1.0.1 = `b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f` | OpenMDW-1.1 | this repository's release | `NOTICE-kumo.txt`, `LICENSE-kumo.txt` |
| `exaone-tabular-classifier-v1_default.safetensors` | EXAONE-Tabular v1 classifier, unmodified | `LG-AI-Research/EXAONE-Tabular` @ `093ad1c2613e1e95d26df8610451d2e02ed53cbe` | EXAONE AI Model License 1.2-NC | Hugging Face only (`scripts/fetch_weights.sh`); not redistributed here | `NOTICE-exa.txt`, `LICENSE-exa-weights.txt` |
| (none, prefetched through `models.txt`) | TabPFN-v2-clf, fallback only | `Prior-Labs/TabPFN-v2-clf` @ `f851f2a3c941544733b712d8c0f96dfae9b28862` | Prior Labs licence (`LICENSE.txt` on HF) | Hugging Face | — |

The modified TabPFN-3.5 checkpoint is not an official product of Prior Labs GmbH. Prior Labs has not endorsed, approved or validated it. Any rights to use the TABPFN-3.5 Model are granted directly by Prior Labs GmbH under its licence. The same applies to the Kumo-Tabular weights and NVIDIA.

## Notices in the archive that describe files not in it

`submission/` is a byte-identical copy of the uploaded archive, so we did not edit the archive's notice files. Four of them are out of date:

- **`NOTICE-v35h.txt`, `NOTICE-v35mh.txt`, `NOTICE-v35h-modification.md` and `NOTICE-v35mh-modification.md`** describe plain bfloat16 recasts of the released TabPFN-3.5 checkpoints (`tabpfn-v3.5-20260909[_multiclass]-bf16.safetensors`). Earlier builds shipped those files; this archive does not. Its only TabPFN-3.5 file is the CPT checkpoint described in `NOTICE-v35-cpt.md`.
- **The two `-modification.md` files** refer to `LICENSE-v35.txt`. The same licence text is in the archive as `LICENSE-v35h.txt` and `LICENSE-v35mh.txt`, which are identical.
