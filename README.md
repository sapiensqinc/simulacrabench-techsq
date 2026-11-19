# SimulacraBench: survey-pretrained tabular foundation models (team `techsq`)

This is the code for team **techsq**'s test-phase submission to the [SimulacraBench](https://simulacrabench.org) competition (NeurIPS 2026 Competition Track). It accompanies our technical report on OpenReview, *SimulacraBench Technical Report: Survey-Pretrained Tabular Foundation Models, Fine-Tuned and Judged Inside the Container*.

The method predicts each `PREDICT` answer's distribution from the `GIVEN` columns with in-context tabular foundation models:

- **Primary:** TabPFN-3.5, continued-pretrained offline on public survey microdata.
- **Partner:** Kumo-Tabular large, continued-pretrained offline and fine-tuned per instrument inside the container.
- **Selection:** the container keeps a component only if it improves a held-out slice.

The report has the full description.

| Codabench Test submission 1055650 | Grand | UNICEF | World Bank | UNHCR |
|---|---|---|---|---|
| skill (leaderboard, 2 decimals) | 0.49 | 0.34 | 0.24 | 0.89 |

## What was submitted

The test-phase upload was `Q35n.zip`, 950,640,304 bytes unpacked, 193 files:

```
sha256(Q35n.zip) = 75edc57f7ecddc0d081e58cb2ee8ab1412803f1c14baa9247bf001c9b0632abf
```

- **`submission/`** is a byte-identical copy of the archive minus its three weight files: 190 files, including `main.py`, `requirements.txt` and `models.txt`.
- **`MANIFEST.sha256`** lists the sha256 of all 193 archive files, weights included.
- **The commit tagged `test-1055650`** is the commit cited in the report.

## Repository layout

```
submission/           the archive's code and notices, byte-identical (entry point: main.py)
MANIFEST.sha256       sha256 of every file in the uploaded archive
scripts/
  fetch_weights.sh    download the three weight files into weights/ and check them
  build_archive.py    submission/ + weights/ -> dist/Q35n-rebuilt.zip, every file checked
  run_local.py        run predict() like the harness and score with the official scorer
cpt/                  continued pre-training of the two modified checkpoints
  README.md           inputs, exact commands, lineage, compute, verification
  run_tabpfn_cpt.sh   run_kumo_cpt.sh   cpt2.py   cpt_kumo.py   ...
  logs/               original training logs
THIRD_PARTY.md        licences and revisions of everything we did not write
LICENSE               Apache-2.0, for our code
```

## Reproducing the submission

### 1. Rebuild the archive

```bash
bash scripts/fetch_weights.sh        # 2 CPT checkpoints (release) + EXAONE (Hugging Face)
python scripts/build_archive.py      # checks all 193 files against MANIFEST.sha256
```

`dist/Q35n-rebuilt.zip` has the same 193 files as the uploaded archive. The zip container bytes differ because of timestamps and compression, so the per-file hashes are the identity check.

### 2. Run it

**Hosted harness.** Upload `dist/Q35n-rebuilt.zip` to the SimulacraBench Test phase on Codabench. This is the only place the test rows exist, because the microdata are not released.

**Contract check.** Run `python tools/check_submission_zip.py dist/Q35n-rebuilt.zip` from a clone of the official kit, <https://github.com/SituatedEvals/public>.

**Local run on sandbox data.** This checks that the pipeline runs end to end under the time budget. It says nothing about the real score.

```bash
git clone https://github.com/SituatedEvals/public kit     # we used 5c69e17
for i in unhcr world_bank unicef; do
  python kit/make_sandbox.py --schema kit/data/$i.json --out sandbox/$i
done
bash scripts/fetch_weights.sh --v2-clf                     # TabPFN-v2-clf into the HF cache
pip install -r submission/requirements.txt
python scripts/run_local.py dist/archive 2 out.json --kit kit --data sandbox
```

`PHASE` sets the wall-clock budget: `1` is the 900 s development budget and `2` is the 3,600 s test budget.

The archive adapts to the clock it measures. It sizes the number of fine-tuning steps and ensemble members from timed steps, so a GPU other than an H100 can give slightly different outputs. The random seed is fixed.

### 3. Rebuild the two modified checkpoints (optional)

See [`cpt/README.md`](cpt/README.md). The run took about 10.3 GPU-hours on one 96 GB GPU: 7.1 h for TabPFN-3.5 and 3.2 h for Kumo.

## Environment

**Hosted image.** Python 3.13 on 1 × H100 with 16 GB RAM and 8 CPU cores. The image pre-installs numpy, pandas, pyarrow, scipy, scikit-learn, torch, huggingface_hub, safetensors and others; the official kit README has the full list. `submission/requirements.txt` adds pinned packages, including `tabpfn==9.0.0` and `skrub==0.10.1`.

**Our local runs.** Python 3.13.16, torch 2.14.1 (CUDA 13.0), numpy 2.5.3 and pandas 3.0.6.

## Models and data

**Pretrained models.** Each model is pinned to a Hugging Face revision; [`THIRD_PARTY.md`](THIRD_PARTY.md) lists them.

- TabPFN-3.5 multiclass @ `06bf2ba3`, plus our CPT
- Kumo-Tabular large classifier v1.0.1 @ `b2f5a9d6`, plus our CPT
- EXAONE-Tabular v1 classifier @ `093ad1c2`
- TabPFN-v2-clf @ `f851f2a3`, fallback only

The base checkpoints remain downloadable at those revisions. This repository's release holds the two modified checkpoints.

**Competition data.** This repository contains no SimulacraBench microdata, fixtures or derived records.

**External data.** It was used only offline, to continue-pretrain the two checkpoints:

- Afrobarometer R1–R8 and R10
- US Census Household Pulse
- NHIS
- OECD PIAAC Cycle 2 and PISA 2018/2022
- Wellcome Global Monitor 2018 (Kumo only)

No external row is shipped in the archive or read at run time. Sources, terms and expected file hashes are in [`cpt/README.md`](cpt/README.md). The report also lists three datasets we used only as offline evaluation fixtures: Afrobarometer R9, GSS, and REACH Iraq MCNA VII. They are not needed to reproduce the submission.

## Licence

Our code is under the Apache License 2.0 (`LICENSE`). Third-party code and weights keep their own licences; see [`THIRD_PARTY.md`](THIRD_PARTY.md). The TabPFN-3.5 and EXAONE weights are licensed for non-commercial use only.
