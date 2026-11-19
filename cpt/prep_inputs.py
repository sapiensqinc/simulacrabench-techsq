"""Afrobarometer .sav -> .pkl, the one input step that was run interactively,
written down as a script on 2026-11-19. With pandas 3.0.6 and pyreadstat
1.3.6 it reproduces all nine pickles in DATA_MANIFEST.sha256 byte for byte.

    python prep_inputs.py   # data/ab_r{8,10}.sav, data/ext/ab_r{1..7}.sav -> .pkl
"""
import os
import pyreadstat

HERE = os.path.dirname(os.path.abspath(__file__))
for r in ("r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r10"):
    d = "data" if r in ("r8", "r9", "r10") else os.path.join("data", "ext")
    sav = os.path.join(HERE, d, "ab_%s.sav" % r)
    if r == "r7":                  # R7 was read as latin-1 although the default also opens it
        df, _ = pyreadstat.read_sav(sav, encoding="latin1")
    else:
        try:
            df, _ = pyreadstat.read_sav(sav)
        except pyreadstat.ReadstatError:   # rounds the default encoding cannot open
            df, _ = pyreadstat.read_sav(sav, encoding="latin1")
    df.to_pickle(sav[:-4] + ".pkl")
    print(r, df.shape)
