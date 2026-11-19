# Continued pre-training (CPT) of the two shipped checkpoints

This directory rebuilds the two checkpoints we changed. The archive ships them as:

| Archive file | From | Steps | sha256 |
|---|---|---|---|
| `cpt_xw_step60000-bf16.ckpt` | TabPFN-3.5 multiclass, `Prior-Labs/tabpfn_3_5` @ `06bf2ba3` | 40,000 + 20,000 | `a11fb013…62c2aa4c` |
| `kumo-tabular-large-classifier-bf16.safetensors` | Kumo-Tabular large classifier, `nvidia/Kumo-Tabular` v1.0.1 @ `b2f5a9d6` | 15,000 + 15,000 + 5,000 | `f72465b4…85d2228e` |

No competition data is used here. All training material is the public survey microdata listed below.

## Lineage

```
TabPFN-3.5 multiclass (fp32, HF 06bf2ba3)
  └─ recast_bf16.py ─► bf16 base (e543f27a…)
       └─ cpt2.py 40k steps, 14 sources, lr 3e-7, seed 5 ─────► runs/cpt_w/step40000.ckpt
            └─ cpt2.py +20k steps, 19 sources, lr 3e-7, seed 9 ─► runs/cpt_xw/step60000.ckpt
                 └─ to_bf16.py ─► cpt_xw_step60000-bf16.ckpt        (archive)

Kumo-Tabular large classifier (HF v1.0.1 = b2f5a9d6)
  └─ cpt_kumo.py 15k steps, 20 sources, lr 1e-6, seed 11 ─► runs/kcpt_a/step15000
       └─ cpt_kumo.py +15k, seed 21 ─────────────────────────► runs/kcpt_a2/step15000
            └─ cpt_kumo.py +5k, 4k–15k-row tasks, seed 31 ──► runs/kcpt_kd/step05000.safetensors (archive)
```

The commands are in `run_tabpfn_cpt.sh` and `run_kumo_cpt.sh`, unchanged from the runs except for relative paths. `logs/` has the original logs. Only the virtual-environment paths in warnings are replaced by `<venv>`. Each Kumo log starts with the per-source sampling weights: the square root of the source's row count, normalised.

## Inputs

Place the downloaded files under `cpt/data/` as below. `DATA_MANIFEST.sha256` gives the hash of every input file we used. Run `cd cpt && sha256sum -c DATA_MANIFEST.sha256` to check them.

| Key(s) | Dataset | Source | Terms | Expected path(s) | Preparation |
|---|---|---|---|---|---|
| `r1`–`r8`, `r10` | Afrobarometer merged data, Rounds 1–8 and 10 | afrobarometer.org | Afrobarometer data policy, citation | `data/ext/ab_r1.sav` … `ab_r7.sav`, `data/ab_r8.sav`, `data/ab_r10.sav` | `python prep_inputs.py` → `.pkl` (reproduces our pickles byte for byte) |
| `hps12`, `hps34`, `hps40` | US Census Household Pulse Survey public-use files, weeks 12, 34, 40 | census.gov | public domain | `data/ext/hps12/pulse2020_puf_12.csv`, `data/ext/hps34/pulse2021_puf_34.csv`, `data/ext/hps40/pulse2021_puf_40.csv` | none |
| `nhis19`–`nhis23` | NHIS sample adult files 2019, 2021, 2022, 2023 | cdc.gov/nchs/nhis | NCHS data use terms | `data/ext/nhis19/adult19.csv` (etc.) | none |
| `piaac22` | OECD PIAAC Cycle 2 public-use files (22 countries) | oecd.org | OECD terms, attribution | `data/ext/piaac/piaac22.pkl` | see the note below |
| `pisa18`, `pisa22` | OECD PISA 2018 and 2022 student questionnaire files | oecd.org/pisa/data | OECD terms | `data/ext/pisa/p18/STU/CY07_MSU_STU_QQQ.sav`, `data/ext/pisa/p22/CY08MSP_STU_QQQ.SAV` | `python make_pisa.py` |
| `wgm18` (Kumo only) | Wellcome Global Monitor 2018 | UK Data Service SN 8466 / wellcome.org | CC BY 4.0 | `data/ext/wgm/wgm2018.parquet`, `data/ext/wgm/dict.csv` | `python make_wgm.py` → `wgm18.csv`, `hold.json` |

`cpt2.py` also lists sources we tried and did not use for these checkpoints (GSS, S-RLS, MCNA, Lebanon return intentions, WGM 2020). The shipped weights did not use them.

Countries held out from training:

- **Afrobarometer:** the 26 countries in `EXCLUDE` at the top of `cpt2.py` are dropped from every round. We used them for checkpoint selection.
- **WGM 2018:** the 20 countries in `hold.json` are dropped. `make_wgm.py` picks them with a fixed seed.

**Note on PIAAC.** We did not keep the script that built `piaac22.pkl`. The file is the PIAAC Cycle 2 public-use files of 22 countries concatenated, 130,114 rows × 1,095 columns, and its sha256 is in `DATA_MANIFEST.sha256`. Likewise, `wgm2018.parquet` and `dict.csv` are the "Full dataset" and "Data dictionary" sheets of the public WGM 2018 workbook, but we did not keep the conversion step. Both hashes are in the manifest.

## Environment

- **Python:** 3.13.16.
- **Packages:** torch 2.14.1 (CUDA 13.0), numpy 2.5.3, pandas 3.0.6, pyreadstat 1.3.6, pyarrow 25.0.1, safetensors 0.8.0, huggingface_hub 2.1.1.
- **TabPFN:** `tabpfn==9.0.0`.
- **Kumo:** NVIDIA structured-data-models @ `842c408fe2a8711bdf2e7cff4bfbe54266d6b940`, identical to `submission/sdm/`. `run_kumo_cpt.sh` puts `submission/` on `PYTHONPATH`. sdm needs torch, numpy, pyarrow, huggingface_hub, safetensors and typing_extensions.
- **Hardware:** one 96 GB GPU.

## Compute (from the logs)

| Run | Steps | Seconds |
|---|---|---|
| `cpt_w` | 40,000 | 17,958 |
| `cpt_xw` | 20,000 | 7,498 |
| **TabPFN total** | **60,000** | **25,456 (7.1 h)** |
| `kcpt_a` | 15,000 | 6,254 |
| `kcpt_a2` | 15,000 | 3,651 |
| `kcpt_kd` (to the step-5,000 save) | 5,000 | 1,764 |
| **Kumo total** | **35,000** | **11,669 (3.2 h)** |

## Verifying a rebuild

- **TabPFN:** `torch.save` output is not byte-deterministic, so compare contents rather than file hashes. Run `python compare_ckpt.py runs/cpt_xw_step60000-bf16.ckpt ../weights/cpt_xw_step60000-bf16.ckpt`. On 2026-11-19, `to_bf16.py` applied to our original step-60,000 checkpoint gave 595/595 identical tensors and identical config, with a different file hash.
- **Kumo:** the original `runs/kcpt_kd/step05000.safetensors` has the same sha256 as the archive file.
- **GPU nondeterminism:** a run from scratch on other hardware or library versions will not be bit-identical, because GPU kernels are not deterministic.
