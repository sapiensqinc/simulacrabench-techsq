"""PISA 2018 / 2022 student questionnaire -> compact CPT source tables.

Per year: up to 2,500 students from each country, questionnaire items (ST*,
IC*, EC*, ...), the derived scale indices (WLEs such as ANXMAT, GROSAGR,
PERSEVAGR: psychometric scale scores like the WB instrument's Big Five /
proactive-personality items), and the first plausible value per domain.
Weights, replicate weights, PV2-10, ids and admin columns are dropped.

    python make_pisa.py   -> data/ext/pisa/pisa18.pkl, pisa22.pkl
"""
import os, re
import numpy as np, pandas as pd, pyreadstat

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {"pisa18": "data/ext/pisa/p18/STU/CY07_MSU_STU_QQQ.sav",
         "pisa22": "data/ext/pisa/p22/CY08MSP_STU_QQQ.SAV"}
DROP = re.compile(r"^(W_|PV([2-9]|10)|CNTSTUID|CNTSCHID|CNTRYID|CYC|NatCen|STRATUM|SUBNATIO|"
                  r"OECD|ADMINMODE|LANGTEST|Option_|VER_DAT|SENWT|BOOKID|TESTLANG|UNIT|EFFORT|"
                  r"REGION|ISCED|.*_D$)", re.I)

for name, path in FILES.items():
    _, meta = pyreadstat.read_sav(os.path.join(HERE, path), metadataonly=True)
    cols = [c for c in meta.column_names if not DROP.match(c)]
    df, _ = pyreadstat.read_sav(os.path.join(HERE, path), usecols=cols,
                                apply_value_formats=False)
    rng = np.random.default_rng(0)
    keep = []
    for _, idx in df.groupby("CNT").indices.items():
        keep.extend(rng.choice(idx, min(2500, len(idx)), replace=False))
    df = df.iloc[np.sort(keep)].reset_index(drop=True)
    df = df[[c for c in df.columns if df[c].notna().mean() > 0.3]]
    df.to_pickle(os.path.join(HERE, "data", "ext", "pisa", name + ".pkl"))
    print(name, df.shape, df["CNT"].nunique())
