"""Wellcome Global Monitor 2018 (CC-BY 4.0, individual level, 149k x 60) ->
data/ext/wgm/wgm18.csv for cpt2.py plus a fixed hold-out country list for
ruler_wgm.py. Derived indices and weights are dropped; missing stays missing
(the CPT loader codes it as its own level)."""
import json, re, numpy as np, pandas as pd
df = pd.read_parquet("data/ext/wgm/wgm2018.parquet")
dd = pd.read_csv("data/ext/wgm/dict.csv")
lab = dict(re.findall(r"(\d+)=([^,]+)", str(dd.iloc[0, 2])))
df = df.drop(columns=["wgt", "PROJWT", "FIELD_DATE", "YEAR_CALENDAR", "WGM_Index",
                      "WGM_Indexr", "ViewOfScience", "Regions_Report", "WBI"])
UNICEF = {"Brazil", "Egypt", "Guatemala", "Iraq", "Kenya", "Mali", "Nepal",
          "Nigeria", "Pakistan", "Zambia"}
codes = sorted(df["WP5"].unique())
names = {int(c): lab.get(str(int(c)), str(int(c))).strip() for c in codes}
pool = [c for c in codes if names[int(c)] not in UNICEF]
rng = np.random.default_rng(2018)
hold = sorted(int(c) for c in rng.choice(pool, 20, replace=False))
df.to_csv("data/ext/wgm/wgm18.csv", index=False)
json.dump({"hold": hold, "names": {h: names[h] for h in hold},
           "unicef_present": sorted(n for n in names.values() if n in UNICEF)},
          open("data/ext/wgm/hold.json", "w"), indent=1)
print(df.shape, "hold rows", int(df["WP5"].isin(hold).sum()))
print(json.load(open("data/ext/wgm/hold.json"))["names"])
print("UNICEF countries in WGM:", json.load(open("data/ext/wgm/hold.json"))["unicef_present"])
