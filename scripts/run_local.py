"""Run the archive the way the hosted harness does, then score it with the
official scorer.

One process calls predict() on UNHCR, then World Bank, then UNICEF, with one
shared clock. Each output is checked and scored with the functions in the
official kit's score.py.

    python scripts/run_local.py ARCHIVE_DIR PHASE OUT.json --kit KIT --data SANDBOX

ARCHIVE_DIR  the unzipped archive (dist/archive after build_archive.py)
PHASE        1 (development budget) or 2 (test budget)
KIT          a clone of https://github.com/SituatedEvals/public
SANDBOX      a directory with unhcr/, world_bank/ and unicef/ made by the
             kit's make_sandbox.py (the real microdata are not released)

Environment: SB_ONLY=unhcr runs the listed instruments only.
"""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("archive")
ap.add_argument("phase", type=int, choices=(1, 2))
ap.add_argument("out")
ap.add_argument("--kit", required=True)
ap.add_argument("--data", required=True)
a = ap.parse_args()

KIT = os.path.abspath(a.kit)
sys.path.insert(0, KIT)
import numpy as np, pandas as pd
import score as S
from make_sandbox import load_config, load_schema

os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
config = load_config(os.path.join(KIT, "config.yml"))
sys.path.insert(0, os.path.abspath(a.archive))
import main

res = {"tabpfn": __import__("tabpfn").__version__}
ONLY = [x for x in os.environ.get("SB_ONLY", "").split(",") if x]
for inst in ("unhcr", "world_bank", "unicef"):
    if ONLY and inst not in ONLY:
        continue
    spath = os.path.join(KIT, "data", inst + ".json")
    schema = load_schema(spath, config)
    resp = S.load_frames(os.path.join(a.data, inst), schema)
    frame, cells, truth = S.sample_rows(schema, resp, a.phase)
    ref = S.marginal_reference(schema, frame, cells, truth, config["scoring"]["floor"])
    # the driver hands over a JSON round trip of the frame
    cols = list(frame.columns)
    rows = [[None if pd.isna(v) else v for v in r] for r in frame.to_numpy(dtype=object)]
    frame = pd.DataFrame(json.loads(json.dumps(rows)), columns=cols)
    with open(spath, encoding="utf-8") as fh:
        raw_schema = json.load(fh)
    raw_schema["gated_value"] = config["gated_value"]
    t = time.time()
    vec = main.predict(frame, raw_schema)
    sec = time.time() - t
    vec = [np.asarray(v, float) for v in vec]
    S.check(schema, vec, cells)
    fl = S.floored(vec, schema, cells, config["scoring"]["floor"])
    sc = S.score(schema, config, fl, truth, cells, ref)
    res[inst] = {"skill": sc["skill"], "log_score": sc["log_score"],
                 "R": sc["marginal_reference"], "sec": round(sec, 1)}
    print("[run_local] %s skill %.4f  %.1fs  wall %.1fs" %
          (inst, sc["skill"], sec, time.time() - main._STARTED), flush=True)
res["wall"] = time.time() - main._STARTED
with open(a.out, "w") as fh:
    json.dump(res, fh, default=str, indent=1)
print(json.dumps({k: res[k] for k in ("tabpfn", "unhcr", "world_bank", "unicef", "wall") if k in res}))
