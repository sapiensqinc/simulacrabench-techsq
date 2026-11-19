"""Continued pretraining of TabPFN-3.5 on many survey tables (Real-TabPFN recipe:
batch 1, tiny constant LR), tabpfn 9.0.0 -- multi-round / multi-family version
of cpt.py (Afrobarometer R1-R8 + R10, optionally GSS and gated targets).

Tasks come from Afrobarometer R8 + R10, never from the countries any R9 ruler
seed (0-2) evaluates on. Each task: a random country subset, 2-8k rows,
GIVEN = COUNTRY, REGION, URBRUR + 4-14 random other columns, target = a random
2-25-level column; every column factorised to integer codes and declared
categorical, as the card does.

    python cpt2.py --ckpt BASE --out DIR --sources r1,r2,...,r10 [--gated 0.3]
"""
import unicodedata
import argparse, json, math, os, time
import numpy as np, pandas as pd, torch

EXCLUDE = {'Angola', 'Botswana', 'Congo-Brazzaville', "Côte d'Ivoire", 'Gambia',
           'Gambia, The', 'Guinea', 'Kenya', 'Liberia', 'Madagascar', 'Malawi',
           'Mali', 'Mauritania', 'Morocco', 'Namibia', 'Nigeria', 'Seychelles',
           'Sierra Leone', 'South Africa', 'Sudan', 'São Tomé and Príncipe',
           'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia'}
ANCHOR = ["COUNTRY", "REGION", "URBRUR"]
GSS_ANCHOR = ["region", "sex", "age"]
GSS_FILES = {"gss18": "data/ext/gss2018/GSS2018.dta",
             "gss22": "data/ext/gss2022/2022/GSS2022.dta",
             "gss24": "data/ext/gss2024/2024/GSS2024.dta"}


def _norm(x):
    try:
        x = str(x).encode("latin1").decode("utf8")     # undo mojibake
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
    return x.lower().replace(",", "").replace("the ", "").strip()


# CSV families: name -> (path, anchors, group column used like COUNTRY)
CSV_SOURCES = {
    "hps34": ("data/ext/hps34/pulse2021_puf_34.csv", ["EST_ST", "EGENID_BIRTH", "TBIRTH_YEAR"], "EST_ST"),
    "hps40": ("data/ext/hps40/pulse2021_puf_40.csv", ["EST_ST", "EGENID_BIRTH", "TBIRTH_YEAR"], "EST_ST"),
    "nhis22": ("data/ext/nhis22/adult22.csv", ["REGION", "SEX_A", "AGEP_A"], None),
    "nhis23": ("data/ext/nhis23/adult23.csv", ["REGION", "SEX_A", "AGEP_A"], None),
    "piaac": ("data/ext/piaac/piaac12.pkl", ["CNTRYID", "GENDER_R", "AGEG10LFS"], "CNTRYID"),
    "piaac22": ("data/ext/piaac/piaac22.pkl", ["CNTRYID", "GENDER_R", "AGEG10LFS"], "CNTRYID"),
    "pisa18": ("data/ext/pisa/pisa18.pkl", ["CNT", "ST004D01T", "ST001D01T"], "CNT"),
    "pisa22": ("data/ext/pisa/pisa22.pkl", ["CNT", "ST004D01T", "ST001D01T"], "CNT"),
    "hps12": ("data/ext/hps12/pulse2020_puf_12.csv", ["EST_ST", "EGENDER", "TBIRTH_YEAR"], "EST_ST"),
    "nhis19": ("data/ext/nhis19/adult19.csv", ["REGION", "SEX_A", "AGEP_A"], None),
    "nhis21": ("data/ext/nhis21/adult21.csv", ["REGION", "SEX_A", "AGEP_A"], None),
    # 10/8: Wellcome Global Monitor 2018 (UNICEF-like: country + vaccine/trust items);
    # the 20 ruler_wgm hold-out countries are dropped below
    "wgm18": ("data/ext/wgm/wgm18.csv", ["WP5", "Gender", "Urban_Rural"], "WP5"),
    # 10/14 (make_ext1014.py): UNHCR-like refugee / displacement household surveys and WGM 2020 (UNICEF-like)
    "srls21": ("data/ext/srls21/srls21.pkl", ["governorate", "country"], None),
    "srls23": ("data/ext/srls23/srls23.pkl", ["Governorat_last_round", "Country_last_round"], None),
    "srls22": ("data/ext/srls22/srls22.pkl", ["gender_numi_1", "age_int_old_1"], None),
    "srls24": ("data/ext/srls24/srls24.pkl", ["Governorat_last_round_value", "age_int_old_1"], None),
    "mcna7": ("data/ext/mcna7/mcna7.pkl", ["governorate_mcna", "population_group"], "governorate_mcna"),
    "mcna8": ("data/ext/mcna8/mcna8.pkl", ["governorate_mcna", "population_group"], "governorate_mcna"),
    "lebret": ("data/ext/lebret/lebret.pkl", ["location_gov", "loc_urban", "head_age"], "location_gov"),
    "wgm20": ("data/ext/wgm20/wgm20.pkl", ["COUNTRYNEW", "Gender", "age_var1"], "COUNTRYNEW"),
}
DROP_PREFIX = ("PWEIGHT", "HWEIGHT", "WTFA", "WTIA", "PPSU", "PSTRAT", "HHX", "SRVY_YR",
               "RECTYPE", "INTV_", "SCRAM", "WEEK", "IMPINC", "POVRATTC", "FAMINCTC",
               "HHSTAT", "ASTATFLG", "PROXY")
EXCLUDE_N = None
HERE = os.path.dirname(os.path.abspath(__file__))


def _codes(s):
    v = s.astype(str).where(s.notna(), "~NA").to_numpy()
    return np.searchsorted(np.unique(v), v).astype(np.float32)


def load_source(r, gated):
    """One table -> integer-coded features/targets. Afro rounds drop every
    country any R9 ruler seed evaluates on; GSS years are whole files."""
    import pyreadstat
    global EXCLUDE_N
    EXCLUDE_N = EXCLUDE_N or {_norm(c) for c in EXCLUDE}
    if r.startswith("gss"):
        df = pd.read_pickle(os.path.join(HERE, "data", "ext", r + ".pkl"))
        anchor, group = GSS_ANCHOR, np.zeros(len(df), int)
    elif r in CSV_SOURCES:
        path, anchor, gcol = CSV_SOURCES[r]
        df = (pd.read_pickle(os.path.join(HERE, path)) if path.endswith(".pkl")
              else pd.read_csv(os.path.join(HERE, path), low_memory=False))
        df = df[[c for c in df.columns if not c.upper().startswith(DROP_PREFIX)]]
        if r == "wgm18":
            hold = json.load(open(os.path.join(HERE, "data/ext/wgm/hold.json")))["hold"]
            df = df[~df["WP5"].isin(hold)].reset_index(drop=True)
        group = df[gcol].to_numpy() if gcol else np.zeros(len(df), int)
    else:
        d = "data" if r in ("r8", "r9", "r10") else os.path.join("data", "ext")
        df = pd.read_pickle(os.path.join(HERE, d, "ab_%s.pkl" % r))
        df.columns = [c.upper() for c in df.columns]
        _, meta = pyreadstat.read_sav(os.path.join(HERE, d, "ab_%s.sav" % r),
                                      metadataonly=True, encoding="latin1")
        labels = {k.upper(): v for k, v in meta.variable_value_labels.items()}
        names = df["COUNTRY"].map(labels["COUNTRY"]).map(_norm)
        df = df[~names.isin(EXCLUDE_N)].reset_index(drop=True)
        anchor = [c for c in ANCHOR if c in df.columns]
        group = df["COUNTRY"].to_numpy()
    feat, targ = [], []
    for c in df.columns:
        if c in anchor or c.upper() in ("RESPNO", "ID", "YEAR") or "OTHER" in c.upper():
            continue
        s = df[c]
        if s.dtype == object and s.nunique() > 25:
            continue                      # free text / ids
        k, nn = s.nunique(dropna=False), s.notna().mean()
        if k < 2 or k > 600 or s.value_counts(dropna=False).iloc[0] / len(s) > 0.97:
            continue
        if nn > 0.9 or (r.startswith("gss") and nn > 0.6) or (r.startswith(("nhis", "piaac", "pisa")) and nn > 0.6):
            feat.append(c)
        if k <= 25 and (nn > 0.95 or (gated and nn > gated)):
            targ.append(c)
    codes = {c: _codes(df[c]) for c in anchor + feat}
    for c in targ:
        codes.setdefault(c, _codes(df[c]))
    cont = {}
    if BINNED:          # WB-style targets: a continuous score cut into quantile bins
        for c in df.columns:
            if c in anchor or c.upper() in ("RESPNO", "ID", "YEAR") or df[c].dtype == object:
                continue
            v = pd.to_numeric(df[c], errors="coerce")
            if v.notna().mean() > 0.4 and v.nunique() > 30:
                cont[c] = v.to_numpy(np.float64)
    return dict(feat=feat, targ=targ, codes=codes, anchor=anchor, cont=cont,
                country=group, n=len(df), name=r)


BINNED = 0.0
ROWS = (2000, 8001)        # task row range (cpt_kumo --rows may widen it)


def _bin(v, k):
    q = np.unique(np.nanquantile(v, np.linspace(0, 1, k + 1)[1:-1]))
    out = np.searchsorted(q, v, side="right").astype(np.int64)
    out[np.isnan(v)] = len(q) + 1
    return np.searchsorted(np.unique(out), out)


def sample_task(R, rng, w):
    cs = np.unique(R["country"])
    if len(cs) > 1:
        pick = rng.choice(cs, rng.integers(min(3, len(cs)), min(15, len(cs)) + 1),
                          replace=False)
        idx = np.flatnonzero(np.isin(R["country"], pick))
    else:
        idx = np.arange(R["n"])
    n = int(rng.integers(*ROWS))
    idx = rng.choice(idx, min(n, len(idx)), replace=False)
    if R["cont"] and rng.random() < BINNED:
        t = str(rng.choice(list(R["cont"])))
        y = _bin(R["cont"][t][idx], int(rng.integers(5, 16)))
    else:
        t = str(rng.choice(R["targ"]))
        y = R["codes"][t][idx].astype(np.int64)
        y = np.searchsorted(np.unique(y), y)
    stem = t.upper()[:6]                 # no near-copies of the target as features
    others = [c for c in R["feat"] if c != t and not c.upper().startswith(stem)]
    g = R["anchor"] + list(rng.choice(others, w - len(R["anchor"]), replace=False))
    X = np.stack([R["codes"][c][idx] for c in g], 1)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--save", default="0,1500,3000,6000")
    ap.add_argument("--lr", type=float, default=3e-7)
    ap.add_argument("--warm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--sources", default="r8,r10")
    ap.add_argument("--binned", type=float, default=0.0,
                    help="share of tasks whose target is a binned continuous column")
    ap.add_argument("--boost", default="",
                    help="source weight multipliers, e.g. wgm18:6")
    ap.add_argument("--gated", type=float, default=0.0,
                    help="also use targets present for > this share, missing = a level")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from functools import partial
    from tabpfn import TabPFNClassifier
    from tabpfn.model_loading import save_tabpfn_model
    from tabpfn.architectures.interface import PerformanceOptions
    from tabpfn.finetuning._torch_compat import GradScaler, autocast, sdpa_kernel_context
    from tabpfn.finetuning.data_util import (get_preprocessed_dataset_chunks,
                                            meta_dataset_collator)
    from tabpfn.finetuning.finetuned_base import _ratio_ctx_query_split
    from torch.utils.data import DataLoader

    global BINNED
    BINNED = a.binned
    rounds = [load_source(r, a.gated) for r in a.sources.split(",")]
    print(json.dumps({r["name"]: dict(rows=r["n"], feat=len(r["feat"]),
                                      targ=len(r["targ"])) for r in rounds}), flush=True)
    pw = np.sqrt([r["n"] for r in rounds])
    for kv in filter(None, a.boost.split(",")):
        k, v = kv.split(":")
        pw = pw * np.array([float(v) if r["name"] == k else 1.0 for r in rounds])
    pw = pw / pw.sum()
    print("source weights", {r["name"]: round(float(p), 3) for r, p in zip(rounds, pw)}, flush=True)
    rng = np.random.default_rng(a.seed)
    clf = TabPFNClassifier(n_estimators=1, device="cuda", model_path=a.ckpt,
                           ignore_pretraining_limits=True, random_state=a.seed,
                           fit_mode="batched", differentiable_input=False,
                           inference_config={"ENABLE_GPU_PREPROCESSING": False})
    clf._initialize_model_variables()
    model = clf.models_[0].float().to("cuda")
    print("param dtype", next(model.parameters()).dtype, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warm))
    scaler = GradScaler()
    perf = PerformanceOptions(force_recompute_layer=True,
                              use_chunkwise_inference=False)
    saves = sorted(int(s) for s in a.save.split(","))

    def save(step):
        model.eval()
        p = os.path.join(a.out, "step%05d.ckpt" % (step + a.offset))
        save_tabpfn_model(clf, p)
        model.train()
        print("saved", p, flush=True)

    step, losses, t0, chunk_i, skipped = 0, [], time.time(), 0, 0
    model.train()
    if 0 in saves:
        save(0)
    while step < a.steps:
        w = int(rng.integers(7, 18))
        tasks = [sample_task(rounds[rng.choice(len(rounds), p=pw)], rng, w)
                 for _ in range(a.chunk)]
        clf.categorical_features_indices = list(range(w))
        seed_e = a.seed * 100000 + chunk_i
        split = partial(_ratio_ctx_query_split, query_ratio=0.2, min_query_size=30,
                        random_state=seed_e)
        ds = get_preprocessed_dataset_chunks(
            calling_instance=clf, X_raw=[t[0] for t in tasks],
            y_raw=[t[1] for t in tasks], split_fn=split, max_data_size=12000,
            model_type="classifier", equal_split_size=False,
            data_shuffle_seed=seed_e, preprocessing_random_state=a.seed)
        loader = DataLoader(ds, batch_size=1, collate_fn=meta_dataset_collator,
                            shuffle=True, generator=torch.Generator().manual_seed(seed_e))
        for batch in loader:
            if step >= a.steps:
                break
            ctx_u = torch.unique(torch.cat([t.reshape(-1) for t in batch.y_context]))
            qry_u = torch.unique(batch.y_query.reshape(-1))
            if not bool(torch.isin(qry_u, ctx_u).all()):
                continue
            n_cls = int(max(ctx_u.max().item(), qry_u.max().item())) + 1
            clf.n_classes_ = n_cls
            clf.classes_ = np.arange(n_cls)
            opt.zero_grad()
            try:
                clf.fit_from_preprocessed(batch.X_context, batch.y_context,
                                          batch.cat_indices, batch.configs,
                                          performance_options=perf)
                with autocast(enabled=True), sdpa_kernel_context():
                    logits = clf.forward(batch.X_query, return_raw_logits=True)
                    Q, B, E, L = logits.shape
                    lg = logits.permute(1, 2, 3, 0).reshape(B * E, L, Q)
                    tgt = batch.y_query.repeat(B * E, 1).to("cuda").long()
                    loss = torch.nn.functional.cross_entropy(lg.float(), tgt)
            except ValueError as exc:      # class-count edge case in a batch
                skipped += 1
                opt.zero_grad()
                continue
            with sdpa_kernel_context():
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            losses.append(float(loss.item()))
            if step % 100 == 0:
                print(json.dumps(dict(step=step, loss100=round(np.mean(losses[-100:]), 4), skipped=skipped,
                                      sec=round(time.time() - t0))), flush=True)
            if step in saves:
                save(step)
        chunk_i += 1
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
