"""One estimator per item, and a ruler that reads the delivered rows.

v17 composes the plan per item. The gate runs each feature view (base, geo
with the place codes replaced by coordinates, round-2-only, ordinal
numbers, roster total) and each second backbone (TabPFN v2.6, v2) once on
the slice; the instrument-wide view is chosen first, then every item may
take the view that beats it by `view_z` bootstrap SEs, then a partner
backbone (or the mean of two) is mixed in instrument-wide and per item.
The final pass runs each named pass only on the items that name it.

The estimator is v15's: TabPFN-3 per item with the hurdle decomposition
(sentinel as its own binary fit, the answer fitted on the asked rows), the
raw GIVEN codes as categorical features, and the round-coded items fitted on
the scored round alone. That is the estimator that reads 0.52 / 0.50 / 0.92
on the delivered rows.

v16 spends the idle clock -- 328 s of 900 -- on the one thing no local
fixture can do: it measures on the real instrument. Before the final fit it
holds out a slice of the visible rows drawn from the same round as the scored
block, fits the candidates on the rest, and scores them on the slice with the
grader's own metric. What the slice decides:

  * the calibration -- one power on the sentinel logits, one on the answer
    vectors, one on the ungated vectors, and the crowd floor -- fitted per
    instrument by coordinate descent on the slice, then applied to the final
    vectors;
  * whether a candidate beats the shipped estimator by more than the accept
    margin: the v2 backbone averaged in (0.25 / 0.5), every item fitted on the
    scored round alone, the scored round duplicated in the context, or the
    household total as an extra column.

Every one of those is a question the fixtures answer wrongly or not at all
(they have no drift between rounds, no ordinal structure, and a cell-lookup
truth that makes any two good backbones agree). The slice is the delivered
population. A candidate that loses on it is never shipped, so the worst case
is the v15 estimator with a calibration fitted to its own real errors.
"""

import hashlib
import os
import re
import sys
import time

import numpy as np
import pandas as pd

_STARTED = time.time()
try:
    with open("/proc/self/stat") as _fh:
        _ticks = float(_fh.read().rsplit(")", 1)[1].split()[19])
    with open("/proc/uptime") as _fh:
        _up = float(_fh.read().split()[0])
    _STARTED = time.time() - (_up - _ticks / os.sysconf("SC_CLK_TCK"))
except Exception:
    pass

CONFIG = {
    "budget": {1: 900.0, 2: 3600.0},
    "safety": 0.92,
    # ---- v22b hardening ------------------------------------------------
    # TabFM loads before the first call that uses it splits its clock, and
    # only when the container can carry it: at least `tfm_min_free_gb` of
    # non-reclaimable memory headroom in the cgroup, and a probe read of the
    # checkpoint that projects the whole 6.5 GB file within `tfm_max_load_s`
    # and within `tfm_load_share` of what is left of the run.
    "tfm_preload": False,
    "tfm_min_free_gb": 6.0,
    "tfm_max_load_s": 150.0,
    "tfm_load_share": 0.25,
    "tfm_probe_mb": 512,
    # If the full pipeline raises, the instrument is predicted again with
    # every optional stage off (the base TabPFN pass), when at least
    # `fallback_min_s` of the run is left; otherwise, or if that raises too,
    # with the crowd marginals of the visible rows.
    "fallback_min_s": 120.0,
    "share": {"unhcr": 0.54, "world_bank": 0.30, "unicef": 0.16},
    "estimators": 8,
    "estimators_sentinel": 8,
    "rows_cap": 20000,
    "max_width": 160,
    "narrow_width": 10,
    "floor": 0.004,
    "smoothing": 0.5,
    "seed": 0,
    "stack_instruments": [],
    "hybrid_instruments": [],
    # v18: the TabPFN-3 multiclass checkpoint is the primary backbone. On
    # the weak-signal World Bank fixture it is +0.0029 over the default v3
    # (narrow items included), on sharp_unhcr_w2 +0.0027, UNICEF +0.0004;
    # no partner adds to it there. The default v3 stays available as the
    # partner `v3d`.
    "backbone": "v3m",
    "wide_backbone": "v3m",
    # v30: per-instrument overrides of the two above (empty = the globals)
    "backbone_by_instrument": {},
    "wide_backbone_by_instrument": {},
    # Optional checkpoint for the two-class fits (sentinel and non-response
    # hurdles); None keeps the primary. `v3b` is TabPFN-3's binary-specialised
    # checkpoint.
    "binary_backbone": None,
    # ---- v18: a GIVEN level the held-out rows carry and the visible rows
    # do not is read as its nearest visible level when a leave-one-level-out
    # test on the visible rows says that beats reading it as missing.
    "unseen_map": True,
    "unseen_share": 0.25,
    "unseen_min_held": 20,
    "unseen_max_vis": 5,
    "unseen_pool_min": 50,
    "unseen_judges": 2,
    "unseen_items": 40,
    "unseen_query_cap": 400,
    "unseen_estimators": 2,
    "unseen_margin": 0.002,
    # ---- v18: items that share one option list (the three sector items,
    # the hours/applications counts, the wage bands) are fitted together
    # in the `pool` view -- one context per family with the member index as
    # a column -- so a 25-class item borrows the others' rows. Families of
    # up to `pool_max_members`, options at least `pool_min_width` wide.
    "pool": True,
    "pool_min_width": 6,
    "pool_max_members": 8,
    "pool_rows_cap": 12000,
    # option lists join one family when their canonical labels overlap by
    # this Jaccard share (the tracer's "Sports" for "Sport", a missing "0")
    "pool_jaccard": 0.75,
    "exact_gates": False,
    "wave_fit": "both",
    "wave_floor": 400,
    "roster_sum": False,
    # ---- v16: the in-container ruler ---------------------------------
    # Hold out `gate_rows` visible respondents from the scored round (or
    # from all visible rows when the scored block is mixed), fit on the
    # rest, score on them, and decide the calibration and the candidates.
    "gate": True,
    "gate_rows": {"unhcr": 600, "world_bank": 600, "unicef": 1000},
    "gate_estimators": 4,
    # The gate may spend at most this fraction of the instrument's slice;
    # the final pass keeps the rest, and drops to `gate_estimators` when
    # what is left would not carry `estimators`.
    "gate_share": {"unhcr": 0.62, "world_bank": 0.65, "unicef": 0.7},
    # Candidates in priority order. Each costs one pass on the slice;
    # `cost` is the expected cost relative to the base pass, used to skip
    # a candidate that would not finish inside the gate's budget.
    # v20: the recode passes first -- they refit 68 items (about 27 s on the
    # real file) and carry the largest fixture gain, so a slow gate must not
    # push them out behind a 70 s backbone pass.
    # v25: the 900 s gate spends its clock on structural hypotheses the
    # fixtures cannot price (parent propagation, the parent-probability
    # chain, per-level fits, one-hot places), not on partners measured at 0
    # on every fixture at this clock (ft, v26, v3d, the CPU family). ft
    # stays a final-phase candidate (`ft_phases`).
    "candidates": ["recode2", "recode", "prop", "chain", "split", "onehot",
                   "ordnum", "geo", "nest", "exa", "ft", "pool", "roster",
                   "delta", "all", "dup2"],
    "ft_phases": [2],
    "onehot_hi_card": 30,
    # prop: a gated item whose parent is a PREDICT item takes its sentinel
    # probability from the parent's own predicted vector, propagated through
    # the empirical routing table P(sentinel | parent level) of the visible
    # rows, instead of a binary fit of its own.
    "prop": True,
    # chain: out-of-fold predicted vectors of the largest PREDICT parents
    # (at least `chain_min_children` children) join the GIVEN codes as
    # numeric columns for every item's fit.
    "chain": True,
    "chain_min_children": 3,
    "chain_max_parents": 6,
    "chain_folds": 2,
    # v38 lat: a latent respondent type read from the OTHER items. EM fits a
    # `lat_k`-class model on the context rows' whole answer matrix; each item
    # is then fitted on (x, k_-j), k_-j the class its row gets from every
    # item but itself, and read back as sum_k p(k | x) p(y | x, k), with
    # p(k | x) one more fit over the GIVEN block. Real-survey ruler
    # (Afrobarometer, 2026-09-29): 50:50 with the per-item fit +0.0007 on the
    # WB shape (31/9 items, z 3.7) and +0.0005 on the UNICEF shape (26/4,
    # z 6.0). Offered to the gate as a view; nothing moves unless the slice
    # says so.
    "lat": True,
    "lat_k": 8,
    "lat_iters": 120,
    "lat_restarts": 2,
    "lat_min_items": 8,
    # the phases that offer it: at 900 s the pass costs the final pass its
    # estimators (afro_unicef phase 1: E8 -> E5, -0.0001) before the slice
    # has the rows to see a +0.0005
    "lat_phases": [1, 2],
    # the instruments that offer it: UNICEF has 12 items, too few for the
    # type to carry anything (afro_unicef phase 2: 6/5 items, -0.00002)
    "lat_instruments": ["unhcr", "world_bank", "unicef"],
    # v38 lat_parent: "replace" | "add" | None -- a gated item with a PREDICT
    # parent integrates over the parent's answer instead of (or besides) the
    # latent class
    "lat_parent": None,
    # the final pass runs lat at this fixed ensemble size (None = the shared E)
    "lat_final_estimators": None,
    # ... and the gate judges it at this ensemble size (None = the gate's)
    "lat_gate_estimators": None,
    # split: every fit is done per level of the GIVEN column that carries
    # the most mutual information with the targets (country of asylum, the
    # survey country), levels with at least `split_min_rows` visible rows;
    # the rest of the rows keep the instrument-wide fit.
    "split": True,
    "split_min_rows": 300,
    "split_prefer": "country",
    # recode: the multi-select families where one round wrote a plain 0 for
    # a skipped respondent. Their parent is a PREDICT item, visible on every
    # training row, so the other round's skips can be rewritten as the
    # sentinel and the item fitted on both rounds instead of the scored one
    # alone (four times the rows for 68 UNHCR items). The slice decides.
    "recode": True,
    "recode_hi": 0.9,
    "recode_lo": 0.1,
    "recode_min": 20,
    # delta: an item that repeats a GIVEN question (the wage bands asked
    # again after the assessment and at the tracer, the main earner's age /
    # relation / education against the respondent's) is fitted as the
    # *shift* from the GIVEN answer, so "the same band" and "one band up"
    # are shared across every starting band instead of being learned per
    # band. Read back per row from the row's own GIVEN value.
    "delta": True,
    # nest: the non-response levels ("Not answered", "Don't know", "Prefer
    # not to answer", 98/99) get their own hurdle inside the answer: a
    # binary fit for "non-response or not", the answer fitted on the rows
    # that answered. The sentinel hurdle bought more than anything else on
    # the delivered rows; this is the same decomposition one level down.
    "nest": True,
    "nest_pattern": r"(not answered|no response|prefer not|don.?t know|dont know|refus|not sure|no answer|not recorded|^9[89]\b|^9[89]\.|\bpna\b)",
    # select: per item, only the GIVEN columns that carry bias-corrected
    # mutual information with it (at least `select_min` columns, columns
    # above `select_frac` of the strongest one). Fewer idle columns, less
    # variance where the signal is weak.
    "select_frac": 0.05,
    "select_min": 2,
    "pass_cost": {"base": 1.0, "geo": 1.2, "all": 0.35, "ordnum": 1.0,
                  "roster": 1.0, "v2": 0.7, "v25": 1.0, "v26": 1.1,
                  "v3b": 1.0, "v3m": 1.0, "v3o": 1.0, "v3d": 1.0, "onehot": 2.0,
                  "select": 0.8, "nest": 1.5, "dup2": 1.25, "pool": 1.0,
                  "recode": 0.5, "recode2": 0.5, "delta": 0.3, "exa": 1.5,
                  "tfm": 1.6, "ft": 1.0, "prop": 0.2, "chain": 1.3,
                  "split": 1.3, "cau": 0.5, "lat": 2.2, "kumo": 1.6},
    # The second backbones that may be mixed into the chosen view, in the
    # order they are tried; singles, pairs and the whole set are tried.
    "mix_partners": ["ft", "exa"],
    # ---- v28: partners forced into every item's plan --------------------
    # On real survey tasks (Afrobarometer, competition shapes; see
    # experiments/survey_prior/RESULTS.md) the mean of the primary with
    # EXAONE Tabular and Causilo beats the primary alone: +0.0010 on the
    # World Bank shape (z 3.3) and +0.0034 on the UNHCR shape (z 8.5),
    # with 35-37 of 39 items improving. The synthetic fixtures priced the
    # same partners at 0, and the slice's verdict on them was a coin the
    # clock rarely reached. So they are no longer judged: after the gate
    # composes the plan, every item a partner can carry (at most 10
    # classes) takes the equal mean of its plan and the partners. The
    # partner passes run last in the final pass; an item the clock cuts
    # keeps its plan vector exactly, and when the clock cannot carry the
    # partner passes at all they are dropped before the primary's E is.
    # v29: back to v26 -- the forced mean lost on every fixture (afro_* and
    # rgu/rgw_real, -0.0002..-0.0014, experiments/survey_prior/RESULTS.md) and
    # the slice judges exa again instead.
    "force_partners": {},
    # 0 = equal weights (each pass 1/(n+1)); otherwise the partners' share
    "force_partner_share": 0.0,
    "cau_max_width": 10,
    "cau_instruments": ["unhcr", "world_bank", "unicef"],
    # v24: the primary backbone fine-tuned inside the container on the
    # instrument's visible rows minus the slice (every item one meta-dataset,
    # shared weights), offered to the gate as one more partner. Fixtures
    # (direct fits, lr 1e-5, 4 passes): World Bank +0.0017, UNHCR +0.0012
    # after 3, UNICEF 0 -- so UNICEF is left out. `ft_share` is the part of
    # the gate's clock the tuning may spend, per phase.
    "ft_instruments": ["world_bank", "unhcr"],
    "ft_lr": 1e-5,
    "ft_epochs": 4,
    "ft_share": {"1": 0.35, "2": 0.4},
    "ft_estimators": 2,
    "ft_query_ratio": 0.2,
    "ft_max_ctx": 12000,
    "ft_min_steps": 40,
    # "full": one task per item on its whole option list; "hurdle": gated
    # items contribute the sentinel-vs-asked binary and the asked-rows answer
    # as two tasks, which is what the passes actually fit
    "ft_tasks": "full",
    # v21: TabFM carries at most 10 classes; wider items keep the primary.
    # On the fixtures it beats v3m directly by +0.004 (WB), +0.0016
    # (UNICEF), +0.019 (hard UNHCR); its pass costs about 3 base passes.
    "tfm_max_width": 10,
    "tfm_estimators": 2,
    "tfm_instruments": [],
    # per instrument, at most this many items go to TabFM (worst by the base
    # pass's slice loss); 0 = every narrow item. UNHCR's 213 items at ~1.5 s
    # each do not fit the 900 s clock next to the other passes.
    "tfm_items_max": {"unhcr": 70},
    # v22: the same rule for every other candidate pass; 0 = all items. The
    # recode passes refit 68 items anyway and stay whole.
    "gate_items_max": {"unhcr": 70, "world_bank": 40, "unicef": 0},
    "full_candidates": ["recode", "recode2", "prop"],
    # v26. Views applied to every pass, base included, and therefore never
    # judged by the gate. Empty keeps v25's behaviour exactly.
    "base_views": [],
    # v19: EXAONE fits items up to this many classes natively; wider items
    # keep the primary vector (its ECOC path is slower and untested here).
    "exa_max_width": 10,
    # EXAONE loses 0.002 on the World Bank fixture and mixes to exactly 0
    # there, and the World Bank gate has the least clock to spare; the pass
    # runs only where it was measured level or better.
    "exa_instruments": ["unhcr", "unicef"],
    # 10/8: Kumo-Tabular large; on the Afrobarometer ruler its 0.2-0.35 mix
    # with the CPT backbone is -0.0004/-0.0005/-0.0004 nats (UNICEF/WB/UNHCR)
    "kumo_instruments": ["unhcr", "world_bank", "unicef"],
    "kumo_max_width": 10,
    # ---- v18: a different family on the CPU, while the GPU passes run ----
    # `cpu_worker.py` fits gradient-boosted trees (hgb) and a one-hot
    # logistic regression with pairwise interactions (lr2) in its own
    # interpreter on the same rows the gate passes use. They predict the
    # slice and the held-out rows in one go, so the final pass reuses them
    # (no refit). The slice decides their weight, instrument-wide and per
    # item; an item the worker did not reach keeps its GPU vector exactly.
    "cpu_passes": ["lr2", "tab2", "hgbs", "hgb", "lr2s"],
    # v26: kNN / hot-deck donors over the GIVEN block. Not another per-item
    # conditional fit -- it is the one family the roster never had.
    "donor": {"k": 60, "tau": 1.0, "prior": 2.0},
    # regularisation variants of the CPU family (the slice picks): hgbs is a
    # shallower, heavier-shrunk tree; lr2s a strongly regularised logistic
    "cpu_variants": {"hgbs": {"max_leaf_nodes": 7, "min_samples_leaf": 40,
                              "l2_regularization": 5.0, "max_iter": 80},
                     "lr2s": {"C": 0.1}, "tab2": {"m": 8.0}},
    # partner plans: every single, then pairs and the whole set among the
    # `partner_top` best singles at w = 0.5
    "partner_top": 3,
    "cpu_instruments": [],
    "cpu_threads": 3,
    "cpu_wait": 20.0,
    "hgb": {"max_iter": 120, "learning_rate": 0.08, "max_leaf_nodes": 15,
            "min_samples_leaf": 20, "l2_regularization": 1.0},
    "lr2_C": 0.5,
    "lr2_pair_max_levels": 12,
    "lr2_min_support": 5,
    # A one-hot feature view for the TabPFN passes, tried when the block is
    # small enough (the codes are read half the time as numbers otherwise).
    "onehot_max": 400,
    # The slice follows the held-out block. After the round restriction,
    # the GIVEN column whose held-out distribution differs most from the
    # visible one (total variation above this) stratifies the draw, so the
    # calibration and the candidates are judged on the population the
    # scored rows come from, whatever the file happens to be sorted by.
    "slice_shift_tv": 1.1,
    # v29: final phase only -- judge on the DEV block (see the slice draw).
    "dev_slice": True,
    "dev_slice_max": 1500,
    "tabpfn_inference": {},
    # geo: the coordinates replace the place columns that TabPFN would
    # otherwise read as alphabetical integers (more than 30 levels).
    "geo_replace": True,
    # Per-item feature view: an item leaves the instrument-wide view only
    # when one alternative beats it on the slice by `view_z` bootstrap SEs.
    "item_view": True,
    "view_z": 2.5,
    # The gate passes also predict the held-out rows. With weight 0 those
    # vectors only stand in where the clock cuts the final pass (an item
    # falls to an E4 fit on 93% of the rows instead of the crowd); a
    # positive weight would average them in, which lost 0.003 on sharp_w2
    # because the round-restricted items are fitted on a third fewer rows.
    "gate_out": True,
    "gate_out_weight": 0.0,
    # A candidate must beat the calibrated base on the slice by this much
    # skill to be shipped; ties keep the base.
    "accept_margin": 0.0015,
    "accept_margin_by_instrument": {},
    # v25: the margin for a feature view, on the items its pass reached
    "view_margin": 0.002,
    # Per-item mixing weight for the v2 backbone: an item leaves the
    # instrument-wide weight only when its own slice gain over that weight
    # exceeds `item_z` respondent-bootstrap standard errors.
    "item_mix": True,
    "item_z": 2.0,
    "mix_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
    "calibrate": True,
    "power_grid": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7],
    "floor_grid": [0.0002, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.006,
                   0.008, 0.012],
    "scorer_floor": 0.001,
    # v46: epsilon de-mixing -- the grader reports eps + (1 - K eps) q, so the
    # q that scores best is (pi - eps) / (1 - K eps), not pi. lambda scales eps;
    # the slice picks it per instrument with the powers and the floor (0 = v43).
    "demix_grid": [0.0, 0.5, 1.0],
}
# Local sweeps (SB_CONFIG) and `package.py --set` both land here.
import json as _json
_OVERRIDE = _json.loads(os.environ.get("SB_CONFIG", "{}"))
# baked in by package.py --set
_OVERRIDE.update({'wide_backbone': 'v35mh', 'base_views': ['recode'], 'backbone_by_instrument': {'unhcr': 'v35h', 'world_bank': 'v35mh', 'unicef': 'v35mh'}, 'wide_backbone_by_instrument': {'unhcr': 'v35h', 'world_bank': 'v35mh', 'unicef': 'v35mh'}, 'ft_instruments': ['world_bank'], 'ft_epochs': 10, 'ft_share': {'1': 0.62, '2': 0.7}, 'full_candidates': ['recode', 'recode2', 'prop', 'ft', 'lat'], 'accept_margin_by_instrument': {'world_bank': 0.0002, 'unhcr': 0.0002, 'unicef': 0.0002}, 'mix_partners': ['ft', 'exa', 'kumo'], 'kumo_ft_instruments_by_phase': {'1': ['world_bank', 'unicef'], '2': ['unhcr', 'world_bank', 'unicef']}, 'kumo_ft_share_by_phase': {'1': {'world_bank': 0.55, 'unicef': 0.7}, '2': {'unhcr': 0.4, 'world_bank': 0.45, 'unicef': 0.7}}, 'kumo_ft_steps_by_phase': {'1': 600, '2': 1200}, 'kumo_ft_lr_by_phase': {'1': 3e-05, '2': 2e-05}, 'ft_instruments_by_phase': {'1': [], '2': ['world_bank']}, 'kumo_max_width': 160, 'candidates': ['recode2', 'recode', 'prop', 'ft', 'lat', 'kumo', 'chain', 'split', 'onehot', 'ordnum', 'geo', 'nest', 'exa', 'pool', 'roster', 'delta', 'all', 'dup2'], 'ft_phases': [1, 2], 'tabpfn_inference': {'world_bank': {'FINGERPRINT_FEATURE': False}, 'unicef': {'FINGERPRINT_FEATURE': False, 'PREPROCESS': 'ordsh', 'FEATURE_SHIFT_METHOD': None, 'CLASS_SHIFT_METHOD': None}}, 'backbone': 'v35mh', 'tabpfn_kwargs': {'unhcr': {'average_before_softmax': True}}, 'ssmax_scale': {'unicef': {'feature_distribution_embedder': 0.8}}, 'lat_phases': [1, 2], 'lat_instruments_by_phase': {'1': ['unhcr', 'world_bank'], '2': ['unhcr', 'world_bank']}, 'lat_parent_by_instrument_by_phase': {'1': {'unhcr': 'only', 'world_bank': 'replace'}, '2': {'unhcr': 'replace'}}, 'lat_gate_estimators_by_phase': {'1': 2}, 'lat_final_estimators_by_phase': {'1': 2}, 'lat_mix_by_phase': {'1': {'grid': [1.0], 'min_gain': 0.0002}, '2': {'grid': [0.5, 0.75, 1.0], 'min_gain': 0.0002}}, 'gate_by_instrument_by_phase': {'1': {}}, 'share_by_phase': {'1': {'unhcr': 0.56, 'world_bank': 0.29, 'unicef': 0.15}}, 'partner_grid_by_instrument_by_phase': {'1': {'world_bank': [0.25, 0.5, 0.75, 1.0]}}, 'lat_mix_by_instrument_by_phase': {'1': {'world_bank': {'grid': [0.5, 0.75, 1.0], 'min_gain': 0.0002}, 'unhcr': {'grid': [1.0], 'min_gain': 0.0}}}, 'gate_share_by_phase': {'1': {'unhcr': 0.62, 'world_bank': 0.72, 'unicef': 0.7}}, 'lat_k_by_phase': {'1': 4}, 'lat_final_estimators_by_instrument_by_phase': {'1': {'unhcr': 4}}, 'gate_estimators_by_instrument_by_phase': {'1': {'unhcr': 2}}})
_OVERRIDE.update({})
_OVERRIDE.update({})
_OVERRIDE.update({})
_OVERRIDE.update({})
_OVERRIDE.update({'skip_passes_by_phase': {'1': {'unhcr': ['split', 'onehot', 'chain', 'ordnum', 'geo', 'nest', 'exa', 'pool', 'lat', 'roster', 'all', 'dup2']}, '2': {'unhcr': ['split', 'onehot', 'chain', 'ordnum', 'geo', 'nest', 'exa', 'pool', 'roster', 'all', 'dup2']}}, 'kumo_ft_instruments_by_phase': {'1': ['unhcr', 'world_bank', 'unicef'], '2': ['unhcr', 'world_bank', 'unicef']}, 'kumo_ft_share_by_phase': {'1': {'unhcr': 0.7, 'world_bank': 0.55, 'unicef': 0.7}, '2': {'unhcr': 0.75, 'world_bank': 0.45, 'unicef': 0.7}}, 'kumo_ft_steps_by_phase': {'1': {'unhcr': 900, 'world_bank': 600, 'unicef': 600}, '2': {'unhcr': 1800, 'world_bank': 1200, 'unicef': 1200}}, 'kumo_ft_lr_by_phase': {'1': 3e-05, '2': {'unhcr': 1.5e-05, 'world_bank': 2e-05, 'unicef': 2e-05}}, 'gate_share_by_phase': {'1': {'unhcr': 0.72, 'world_bank': 0.72, 'unicef': 0.7}}})
_OVERRIDE.update({'share_by_phase': {'1': {'unhcr': 0.56, 'world_bank': 0.25, 'unicef': 0.19}}, 'skip_passes_by_phase': {'1': {'unhcr': ['split', 'onehot', 'chain', 'ordnum', 'geo', 'nest', 'exa', 'pool', 'lat', 'roster', 'all', 'dup2'], 'world_bank': ['split','onehot','chain','ordnum','geo','nest','exa','pool','lat','roster','all','dup2','delta','prop','recode2'], 'unicef': ['split','onehot','chain','ordnum','geo','nest','exa','pool','lat','roster','all','dup2','delta','prop','recode2']}, '2': {'unhcr': ['split', 'onehot', 'chain', 'ordnum', 'geo', 'nest', 'exa', 'pool', 'roster', 'all', 'dup2']}}, 'kumo_ft_share_by_phase': {'1': {'unhcr': 0.7, 'world_bank': 0.55, 'unicef': 0.85}, '2': {'unhcr': 0.75, 'world_bank': 0.45, 'unicef': 0.7}}, 'kumo_ft_steps_by_phase': {'1': {'unhcr': 900, 'world_bank': 600, 'unicef': 1000}, '2': {'unhcr': 1800, 'world_bank': 1200, 'unicef': 1200}}})  # card_U: phase-1 views WB/UNICEF never adopt are off; freed clock goes to UNICEF Kumo ft
_OVERRIDE.update({'gate_share_by_phase': {'1': {'unhcr': 0.72, 'world_bank': 0.72, 'unicef': 0.88}}, 'kumo_ft_share_by_phase': {'1': {'unhcr': 0.7, 'world_bank': 0.55, 'unicef': 0.92}, '2': {'unhcr': 0.75, 'world_bank': 0.45, 'unicef': 0.7}}, 'partner_grid_by_instrument_by_phase': {'1': {'world_bank': [0.25, 0.5, 0.75, 1.0], 'unicef': [0.25, 0.5, 0.75, 0.85, 0.95]}}})  # card_U3: UNICEF ft window = the gate window the skipped views left empty; Kumo mix grid past 0.75
_OVERRIDE.update({'kumo_ft_reserve_by_phase': {'1': {'unicef': 30.0}}})  # card_U4: U3 + 30 s of the UNICEF gate kept for the Kumo pass
_OVERRIDE.update({'lat_parent_by_instrument_by_phase': {'1': {'unhcr': 'only', 'world_bank': 'replace'}, '2': {'unhcr': 'replace', 'world_bank': 'replace'}}})  # card_F: U4 (phase 1, development board) + Swb (phase 2, WB parent integration) in one card
_OVERRIDE.update({'kumo_ft_qfrac': 0.35})  # card_Q: Kumo ft query share 0.35 (ruler: 0.35 beats 0.2 on all 9 shape x seed)
CONFIG.update(_OVERRIDE)
CONFIG["budget"] = {int(k): float(v) for k, v in CONFIG["budget"].items()}
_SEEN = set()
_SPECS = {}
_REPORT = {}
_TRIALS = {}


_TERRA = {"scale": {}, "installed": False}


def _terraform_install():
    """v36: terraform the loaded TabPFN-3.5 instead of attaching to it. Every
    SoftmaxScalingMLP (the learned length-dependent attention scale, "SSMax") is
    tagged at model construction with the top-level stage it belongs to, and its
    output is multiplied by the factor CONFIG["ssmax_scale"][instrument][stage]
    names. Tagging at construction covers every call path (the embedder's layers
    are also run directly by the chunked path) and every construction site.
    A no-op when no instrument names a factor or the build has no TabPFN-3.5."""
    if _TERRA["installed"] or not CONFIG.get("ssmax_scale"):
        return
    try:
        from tabpfn.architectures import tabpfn_v3_5 as arch
    except Exception:
        return
    init0, fwd0 = arch.TabPFNV3p5.__init__, arch.SoftmaxScalingMLP.forward

    def init(self, *args, **kwargs):
        init0(self, *args, **kwargs)
        for name, mod in self.named_modules():
            if isinstance(mod, arch.SoftmaxScalingMLP):
                mod._terra_stage = name.split(".")[0]

    def fwd(self, q, n):
        out = fwd0(self, q, n)
        t = _TERRA["scale"].get(getattr(self, "_terra_stage", None))
        if t is not None:
            _TERRA["calls"] = _TERRA.get("calls", 0) + 1
            return out * t
        return out

    arch.TabPFNV3p5.__init__ = init
    arch.SoftmaxScalingMLP.forward = fwd
    _TERRA["installed"] = True


def _terraform(key):
    """Select this instrument's SSMax factors (empty = the model as trained)."""
    _TERRA["scale"] = {k: float(v) for k, v in
                       ((CONFIG.get("ssmax_scale") or {}).get(key) or {}).items()}
    _TERRA["calls"] = 0


def _clf_kwargs(key):
    """v36: per-instrument TabPFNClassifier arguments beyond the inference config
    (e.g. average_before_softmax), CONFIG["tabpfn_kwargs"][instrument]."""
    return dict((CONFIG.get("tabpfn_kwargs") or {}).get(key) or {})


def _inference_override(key):
    """v34: per-instrument TabPFN inference-config overrides, laid over the
    checkpoint's stored config (a dict names only what changes). `PREPROCESS`
    takes a recipe name, since a JSON config cannot carry PreprocessorConfig.
    Priced on the real-survey ruler and at the final-phase clock (v33 RESULTS)."""
    ov = dict((CONFIG.get("tabpfn_inference") or {}).get(key) or {})
    recipe = ov.pop("PREPROCESS", None)
    if recipe == "ordsh":
        from tabpfn.preprocessing.configs import PreprocessorConfig
        ov["PREPROCESS_TRANSFORMS"] = [PreprocessorConfig(
            "none", categorical_name="ordinal_shuffled",
            max_features_per_estimator=768)]
    return ov or None


def _spec(path):
    if path in _SPECS:
        return _SPECS[path] or None
    _SPECS[path] = False
    try:
        from tabpfn.base import initialize_tabpfn_model, ClassifierModelSpecs
        models, configs, _, inference = initialize_tabpfn_model(
            model_path=path, which="classifier", fit_mode="fit_preprocessors")
        _SPECS[path] = ClassifierModelSpecs(model=models[0],
                                            architecture_config=configs[0],
                                            inference_config=inference)
    except Exception:
        _SPECS[path] = False
    return _SPECS[path] or None


BACKBONES = {
    'v2': ('Prior-Labs/TabPFN-v2-clf', 'tabpfn-v2-classifier-v2_default.ckpt'),
    'v35mh': ('Prior-Labs/tabpfn_3_5', 'cpt_xw_step60000-bf16.ckpt'),
    'exa': ('LG-AI-Research/EXAONE-Tabular', 'exaone-tabular-classifier-v1_default.safetensors'),
    'kumo': ('nvidia/Kumo-Tabular', 'kumo-tabular-large-classifier-bf16.safetensors'),
    'v35h': ('Prior-Labs/tabpfn_3_5', 'cpt_xw_step60000-bf16.ckpt'),
}
# pass name -> backbone name, for the second-backbone passes
PARTNER_BACKBONE = {"v2": "v2", "v25": "v2.5", "v26": "v2.6", "ft": "ft",
                    "v3b": "v3b", "v3m": "v3m", "v3o": "v3o", "v3d": "v3d",
                    "exa": "exa", "tfm": "tfm", "cau": "cau", "kumo": "kumo",
                    "v35mh": "v35mh", "v35h": "v35h"}


def _checkpoint_name(which):
    if which not in BACKBONES:
        return None
    repo, name = BACKBONES[which]
    bundled = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if os.path.exists(bundled):
        return bundled
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo, name, local_files_only=True)
    except Exception:
        pass
    import glob
    for root in (os.environ.get("HF_HOME", ""),
                 os.environ.get("HF_HUB_CACHE", ""),
                 os.path.expanduser("~/.cache/huggingface")):
        if root:
            found = glob.glob(os.path.join(root, "**", name), recursive=True)
            if found:
                return found[0]
    if which not in ("v2", "exa", "tfm", "kumo") and "v2" in BACKBONES:
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(*BACKBONES["v2"], local_files_only=True)
        except Exception:
            return None
    return None


_EXA = {}


def _exa_ok():
    """True when the vendored exaonetabular package imports."""
    if "ok" not in _EXA:
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            if here not in sys.path:
                sys.path.insert(0, here)
            import exaonetabular  # noqa: F401
            _EXA["ok"] = True
        except Exception:
            _EXA["ok"] = False
    return _EXA["ok"]


def _exa_fit(path, Xa, ya, estimators, seed):
    """An EXAONE Tabular classifier fitted on (Xa, ya). The network loads once
    per checkpoint; each fit only stores its support rows and ensemble plan."""
    from exaonetabular import EXAONETabularClassifier
    if path not in _EXA:
        _EXA[path] = EXAONETabularClassifier.from_pretrained(
            weights=path, device="cuda")
    base = _EXA[path]
    manifest = base.manifest.with_overrides(
        ensemble_count=int(estimators), compute_dtype=None, seed=int(seed))
    clf = EXAONETabularClassifier(manifest, device="cuda", model=base.model)
    clf.fit(np.asarray(Xa, np.float32), np.asarray(ya))
    return clf


_KUMO = {}


def _kumo_ok():
    """True when the vendored sdm package (NVIDIA structured-data-models,
    Apache-2.0) imports."""
    if "ok" not in _KUMO:
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            if here not in sys.path:
                sys.path.insert(0, here)
            import sdm  # noqa: F401
            _KUMO["ok"] = True
        except Exception:
            _KUMO["ok"] = False
    return _KUMO["ok"]


class _KumoClf:
    """Kumo-Tabular large (in-context), fitted = its support rows. GIVEN codes
    go in as numerical columns, as on the real-survey ruler."""

    def __init__(self, path, Xa, ya, estimators, seed):
        import torch
        import sdm
        if path not in _KUMO:
            from safetensors.torch import load_file
            m = sdm.models.KumoTabular(task="classification", size="large",
                                       pretrained=False,
                                       device=torch.device("cuda"))
            sd = {k: (v.float() if v.is_floating_point() else v)
                  for k, v in load_file(path, device="cuda").items()}
            next(iter(m.models.values())).load_state_dict(sd)
            m.eval()
            _KUMO[path] = m
            _KUMO["base_sd:" + path] = {k: v.detach().cpu().clone() for k, v in sd.items()}
        self.m, self.est, self.seed = _KUMO[path], int(estimators), int(seed)
        self.Xa = np.asarray(Xa, np.float32)
        self.ya = np.asarray(ya)
        self.classes_ = np.unique(self.ya)

    def _table(self, X):
        import pandas as pd
        import torch
        import sdm
        cols = ["g%d" % j for j in range(X.shape[1])]
        return sdm.TableTensor.from_pandas(
            df=pd.DataFrame(X, columns=cols), stypes={c: "numerical" for c in cols},
            device=torch.device("cuda"))

    def predict_proba(self, Xq):
        import pandas as pd
        import torch
        import sdm
        Xq = np.asarray(Xq, np.float32)
        yc = sdm.TableTensor.from_pandas(
            df=pd.DataFrame({"t": self.ya.astype(np.int64).astype(str)}),
            stypes={"t": "categorical"}, device=torch.device("cuda"))
        gen = torch.Generator(device="cuda").manual_seed(self.seed)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            out = self.m(x_context=self._table(self.Xa), y_context=yc,
                         x_query=self._table(Xq), num_estimators=self.est,
                         generator=gen)
        df = out.to_pandas()
        pos = {int(c): i for i, c in enumerate(self.classes_)}
        pr = np.zeros((len(Xq), len(self.classes_)))
        for c in df.columns:
            pr[:, pos[int(c)]] = df[c].to_numpy()
        return pr / pr.sum(1, keepdims=True)


def _kumo_fit(path, Xa, ya, estimators, seed):
    return _KumoClf(path, Xa, ya, estimators, seed)


def _kumo_finetune(path, X, ys, stop_at, seed, say, lr=None, cap=None):
    """Fine-tune the shared Kumo model in place on this instrument's rows:
    each step one item, 80/20 context/query split of the given rows, CE on
    the query. Starts from the base weights; the schedule (warmup, cosine to
    0) is sized from five timed steps to end by `stop_at` or `kumo_ft_steps`."""
    import pandas as pd
    import torch
    import torch.nn.functional as F
    import sdm
    import sdm.processing as sp
    _KumoClf(path, X[:50], ys[0][:50], 1, seed)          # load once
    m = _KUMO[path]
    inner = next(iter(m.models.values()))
    inner.load_state_dict({k: v.to("cuda") for k, v in _KUMO["base_sd:" + path].items()})
    items = [j for j, y in enumerate(ys) if 2 <= np.unique(y).size <= 10]
    if not items:
        return {"steps": 0}
    recipe = m.default_recipe()
    recipe.target = sp.StypeDispatch(categorical=[sp.AlignCategories(),
                                                  sp.ShuffleCategories(method="shift")],
                                     numerical=sp.Standardize())
    recipe.output = sp.Identity()
    lr = float(lr if lr is not None else CONFIG.get("kumo_ft_lr", 1e-5))
    cap = int(cap if cap is not None else CONFIG.get("kumo_ft_steps", 600))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float32)
    cols = ["g%d" % j for j in range(X.shape[1])]
    dev = torch.device("cuda")
    tab = lambda A: sdm.TableTensor.from_pandas(df=pd.DataFrame(A, columns=cols),
                                                stypes={c: "numerical" for c in cols}, device=dev)
    nq = max(30, int(float(CONFIG.get("kumo_ft_qfrac", 0.2)) * len(X)))
    plan = {"total": cap}
    t0, losses, step, tries = time.time(), [], 0, 0
    m.train()
    while step < plan["total"] and tries < 4 * cap:
        tries += 1
        if step == 5:
            per = (time.time() - t0) / 5.0
            fit = int(max(0.0, stop_at - time.time()) / max(per, 1e-3))
            plan["total"] = min(cap, 5 + fit)
            plan["per_step"] = round(per, 3)
            if plan["total"] < int(CONFIG.get("kumo_ft_min_steps", 100)):
                break
        if time.time() > stop_at:
            break
        j = int(rng.choice(items))
        y = np.asarray(ys[j])
        perm = rng.permutation(len(y))
        q, c = perm[:nq], perm[nq:]
        if not np.isin(y[q], np.unique(y[c])).all():
            continue
        frac = step / max(plan["total"], 1)
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 20.0) * 0.5 * (1 + np.cos(np.pi * frac))
        yc = sdm.TableTensor.from_pandas(df=pd.DataFrame({"t": y[c].astype(np.int64).astype(str)}),
                                         stypes={"t": "categorical"}, device=dev)
        yq = sdm.TableTensor.from_pandas(df=pd.DataFrame({"t": y[q].astype(np.int64).astype(str)}),
                                         stypes={"t": "categorical"}, device=dev)
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = m(x_context=tab(X[c]), y_context=yc, x_query=tab(X[q]), recipe=recipe)[0]
        logits, yi = sdm.evaluation.to_class_indices(out, yq, missing_score=-torch.inf)
        loss = F.cross_entropy(logits.float(), yi)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        step += 1
        losses.append(float(loss.detach()))
    m.eval()
    del opt
    torch.cuda.empty_cache()
    used = step >= int(CONFIG.get("kumo_ft_min_steps", 100))
    _KUMO["tuned"] = used
    if not used:     # too few steps to be worth it: back to the base weights
        inner.load_state_dict({k: v.to("cuda") for k, v in _KUMO["base_sd:" + path].items()})
    return {"steps": step, "planned": plan.get("total"), "per_step": plan.get("per_step"),
            "items": len(items), "used": used, "seconds": round(time.time() - t0, 1),
            "loss_first": round(float(np.mean(losses[:10])), 4) if losses else None,
            "loss_last": round(float(np.mean(losses[-10:])), 4) if losses else None}


_CAU = {}


def _cau_ok():
    """True when the causilo package imports (bundled or installed) and the
    weights sit beside config.json where BACKBONES says."""
    if "ok" in _CAU:
        return _CAU["ok"]
    _CAU["ok"] = False
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        import causilo  # noqa: F401
        from causilo import checkpoints as _ck
        path = _checkpoint_name("cau")
        if path and os.path.exists(path):
            from pathlib import Path
            folder = Path(os.path.dirname(path))
            if (folder / "config.json").exists():
                # the bundled weights, never the hub (the container is offline)
                _ck.load_pretrained_model = lambda task, _f=folder: _ck.load_checkpoint(_f)
                _CAU["dir"] = folder
                _CAU["ok"] = True
    except Exception:
        _CAU["ok"] = False
    return _CAU["ok"]


def _cau_fit(Xa, ya, estimators, seed):
    """A Causilo classifier fitted on (Xa, ya); the network loads once."""
    from causilo import CausiloClassifier
    clf = CausiloClassifier(n_estimators=int(estimators), random_state=int(seed),
                            device="cuda")
    clf.fit(np.asarray(Xa, np.float32), np.asarray(ya))
    return clf


_TFM = {}


def _tfm_ok():
    """True when the tabfm package imports and the prefetched snapshot is in
    the hub cache (models.txt names the repository)."""
    if "ok" in _TFM:
        return _TFM["ok"]
    _TFM["ok"] = False
    try:
        import tabfm  # noqa: F401
        from tabfm import tabfm_v1_0_0_pytorch  # noqa: F401
        path = _checkpoint_name("tfm")
        if path and os.path.exists(path):
            _TFM["snapshot"] = os.path.dirname(os.path.dirname(path))
            _TFM["ok"] = True
    except Exception:
        _TFM["ok"] = False
    return _TFM["ok"]


def _tfm_model():
    """The TabFM network, built on the GPU in bfloat16 and filled one tensor
    at a time from the safetensors file: neither the 6.5 GB float32
    checkpoint nor a float32 copy of the network ever exists (the container
    has 16 GB of host RAM; the mixin's default path peaked at 15.3 GB, and a
    float32 build on the GPU needs 10 GB transiently)."""
    if "model" not in _TFM:
        import json as _j
        import torch
        from safetensors import safe_open
        from tabfm.src.pytorch.tabfm_v1_0_0 import TabFM_HF
        sub_ = os.path.join(_TFM["snapshot"], "classification")
        with open(os.path.join(sub_, "config.json")) as fh:
            cfg = _j.load(fh)
        kwargs = {"is_classifier": cfg.pop("task", "classification")
                  == "classification"}
        for k_ in ("model_type", "version", "framework"):
            cfg.pop(k_, None)
        kwargs.update(cfg)
        prev = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda"):
                model = TabFM_HF(**kwargs)
        finally:
            torch.set_default_dtype(prev)
        model = model.to(device="cuda", dtype=torch.bfloat16)
        state = dict(model.named_parameters())
        state.update(dict(model.named_buffers()))
        with torch.no_grad(), safe_open(os.path.join(sub_, "model.safetensors"),
                                        framework="pt", device="cuda") as fh:
            keys = list(fh.keys())
            missing = [k for k in state if k not in keys]
            if missing:
                raise RuntimeError("tabfm checkpoint lacks %d tensors" % len(missing))
            for k in keys:
                if k in state:
                    state[k].copy_(fh.get_tensor(k).to(state[k].dtype))
        _TFM["model"] = model.eval()
    return _TFM["model"]




def _tfm_fit(Xa, ya, estimators, seed):
    from tabfm import TabFMClassifier
    clf = TabFMClassifier(model=_tfm_model(), n_estimators=int(estimators),
                          random_state=int(seed))
    clf.fit(np.asarray(Xa, np.float32), np.asarray(ya))
    return clf


def _cgroup_memory():
    """(limit, non-reclaimable use) of this container's memory cgroup in
    bytes, or (None, None) when there is no finite limit to read."""
    for lim_p, cur_p, stat_p, key_ in (
            ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current",
             "/sys/fs/cgroup/memory.stat", "inactive_file"),
            ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
             "/sys/fs/cgroup/memory/memory.usage_in_bytes",
             "/sys/fs/cgroup/memory/memory.stat", "total_inactive_file")):
        try:
            with open(lim_p) as fh:
                raw = fh.read().strip()
            if raw == "max" or int(raw) > (1 << 50):
                return None, None
            with open(cur_p) as fh:
                used = int(fh.read().strip())
            try:
                with open(stat_p) as fh:
                    for line in fh:
                        k_, v_ = line.split()[:2]
                        if k_ == key_:
                            used -= int(v_)
                            break
            except Exception:
                pass
            return int(raw), max(used, 0)
        except Exception:
            continue
    return None, None


def _tfm_guard(phase):
    """(ok, note): whether loading TabFM is safe for memory and clock."""
    limit, used = _cgroup_memory()
    if limit:
        free = (limit - used) / float(1 << 30)
        if free < float(CONFIG.get("tfm_min_free_gb", 6.0)):
            return False, "memory headroom %.1f GB" % free
    ckpt = _checkpoint_name("tfm")
    size = os.path.getsize(ckpt)
    probe = min(size, int(CONFIG.get("tfm_probe_mb", 512)) << 20)
    t0 = time.time()
    got = 0
    with open(ckpt, "rb", buffering=0) as fh:
        while got < probe:
            chunk = fh.read(min(probe - got, 16 << 20))
            if not chunk:
                break
            got += len(chunk)
    rate = got / max(time.time() - t0, 1e-3)
    projected = size / max(rate, 1.0)
    left = _STARTED + CONFIG["budget"][phase] * CONFIG["safety"] - time.time()
    note = "read %.0f MB/s, projected load %.0fs, %.0fs of the run left" % (
        rate / 1e6, projected, left)
    if projected > float(CONFIG.get("tfm_max_load_s", 150.0)) or \
            projected > float(CONFIG.get("tfm_load_share", 0.25)) * left:
        return False, note
    return True, note


def _tfm_preload(phase):
    """Load TabFM now, or switch it off for the rest of the run."""
    t0 = time.time()
    try:
        if not (_cuda() and _tfm_ok()):
            _TFM["load_note"] = "unavailable"
        else:
            ok, note = _tfm_guard(phase)
            if not ok:
                _TFM["ok"] = False
                _TFM["load_note"] = "skipped: " + note
            else:
                _tfm_model()
                _TFM["load_note"] = "loaded in %.1fs (%s)" % (
                    time.time() - t0, note)
    except Exception as exc:
        _TFM["ok"] = False
        _TFM.pop("model", None)
        _TFM["load_note"] = "failed: %r" % (exc,)
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    if os.environ.get("SB_DEBUG"):
        print("[debug] tfm preload: %s" % _TFM["load_note"], flush=True)


def _cuda():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# Load once at import when the prefetched snapshot is there and the container
# can carry it (v22b: through the memory-headroom and read-speed guard; the
# phase is not known yet, so the guard measures against the development
# budget, the tighter of the two); any failure leaves TabFM switched off.
if CONFIG.get("tfm_preload"):
    _tfm_preload(1)


_SCALE = (
    (re.compile(r"^\s*(\d+)\s*-\s*(\d+)"), lambda m: (float(m.group(1))
                                                        + float(m.group(2))) / 2),
    (re.compile(r"^\s*(\d+)\s*\+"), lambda m: float(m.group(1)) + 1),
    (re.compile(r"^\s*(\d+)\s*[\.\)]"), lambda m: float(m.group(1))),
    (re.compile(r"^\s*(\d+)\s*$"), lambda m: float(m.group(1))),
)


def _scalar(value):
    text = str(value)
    for pattern, read in _SCALE:
        found = pattern.match(text)
        if found:
            return read(found)
    return None


def _ordered(values):
    read = [_scalar(v) for v in values]
    if sum(x is not None for x in read) < max(2, len(read) - 1):
        return None
    seen = [x for x in read if x is not None]
    if len(set(seen)) != len(seen) or seen != sorted(seen):
        return None
    return read


def _mi(x, y, nx, ny):
    joint = np.zeros((nx, ny))
    np.add.at(joint, (x, y), 1.0)
    joint /= max(joint.sum(), 1.0)
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = joint * np.log(joint / (px * py))
    return float(np.nansum(term))


# Approximate centroids (latitude, longitude) of the governorates the UNHCR
# schema names, and of the Syrian governorates of origin. World knowledge
# about the GIVEN block, nothing read from the data: two categorical codes
# become a place, and a pair of places becomes a distance home.
_GEO = {
    "Ajloun": (32.33, 35.75), "Al Aqaba": (29.53, 35.01), "Al Balqa": (32.04, 35.73),
    "Al Karak": (31.18, 35.70), "Al Mafraq": (32.34, 36.21), "Al Tafilah": (30.84, 35.60),
    "Amman": (31.95, 35.93), "Irbid": (32.56, 35.85), "Jarash": (32.28, 35.90),
    "Maan": (30.19, 35.73), "Madaba": (31.72, 35.79), "Zarqa": (32.07, 36.09),
    "Akkar": (34.55, 36.10), "Beirut": (33.89, 35.50), "Bekaa": (33.85, 35.90),
    "El Nabatieh": (33.38, 35.48), "Hermel/Baalbek": (34.15, 36.25),
    "Mount Lebanon": (33.85, 35.65), "North": (34.43, 35.85), "South Lebanon": (33.45, 35.35),
    "Al-Najaf": (32.03, 44.33), "Al-Sulaymaniyah": (35.56, 45.43), "Duhok": (36.87, 42.99),
    "Erbil": (36.19, 44.01), "Kirkuk": (35.47, 44.39),
    "Alexandria": (31.20, 29.92), "Assiut": (27.18, 31.18), "Aswan": (24.09, 32.90),
    "Behera": (30.85, 30.35), "Beni Suef": (29.07, 31.10), "Cairo": (30.04, 31.24),
    "Dakahlia": (31.04, 31.38), "Damietta": (31.42, 31.81), "Fayoum": (29.31, 30.84),
    "Gharbia": (30.87, 31.00), "Giza": (30.01, 31.21), "Ismailia": (30.60, 32.27),
    "Kafr El-Shikh": (31.11, 30.94), "Kalyoubia": (30.41, 31.21), "Luxor": (25.69, 32.64),
    "Matrouh": (31.35, 27.24), "Menia": (28.11, 30.75), "Menoufia": (30.56, 31.01),
    "Port Said": (31.26, 32.30), "Red Sea": (26.50, 33.50), "Sharkia": (30.59, 31.50),
    "Al-Hasakeh": (36.50, 40.75), "Aleppo": (36.20, 37.16), "Ar-Raqqa": (35.95, 39.01),
    "As-Sweida": (32.71, 36.57), "Damascus": (33.51, 36.29), "Dar'a": (32.62, 36.10),
    "Deir-ez-Zor": (35.34, 40.14), "Hama": (35.13, 36.75), "Homs": (34.73, 36.72),
    "Idleb": (35.93, 36.63), "Lattakia": (35.52, 35.79), "Quneitra": (33.13, 35.82),
    "Rural Damascus": (33.55, 36.50), "Tartous": (34.89, 35.89),
}


# Which surveyed country an unsurveyed one most resembles, by region, majority
# religion and income band -- world knowledge, used only as a *candidate*
# neighbour that the leave-one-level-out judge has to confirm on the delivered
# rows before it is applied to anything.
_KIN = {
    "Pakistan": ["Iraq", "Egypt", "Nepal", "Mali", "Nigeria"],
    "Zambia": ["Kenya", "Nigeria", "Mali", "Guatemala"],
    "Nigeria": ["Kenya", "Mali", "Zambia", "Pakistan"],
    "Nepal": ["Pakistan", "Kenya", "Guatemala", "Mali"],
    "Mali": ["Nigeria", "Egypt", "Kenya", "Pakistan"],
    "Kenya": ["Zambia", "Nigeria", "Mali", "Nepal"],
    "Iraq": ["Egypt", "Pakistan", "Mali", "Nigeria"],
    "Egypt": ["Iraq", "Pakistan", "Mali", "Nigeria"],
    "Guatemala": ["Brazil", "Nepal", "Kenya", "Zambia"],
    "Brazil": ["Guatemala", "Kenya", "Zambia", "Nepal"],
    "Lebanon": ["Jordan", "Iraq", "Egypt"], "Jordan": ["Lebanon", "Iraq", "Egypt"],
    "South Africa": ["Lesotho", "Nigeria"], "Lesotho": ["South Africa", "Nigeria"],
}


# PREDICT items that repeat a GIVEN question, for the delta view.
_TWINS = {
    "user.wageOfSimilarEmployeesAfter": "user.wageOfSimilarEmployees",
    "user.lowestAcceptablePayAfter": "user.lowestAcceptablePay",
    "tr_similar_wage_after": "user.wageOfSimilarEmployees",
    "tr_low_pay_after": "user.lowestAcceptablePay",
    "tr_employment_status": "user.employmentStatus",
    "s4_4_main_earner_age": "s2_1b_age",
    "s4_5_main_earner_relation_lbl": "s2_1_hh_relationship_lbl",
    "s4_6_main_earner_education_lbl": "s2_1c_education_level_lbl",
    "s3_10c_returnee_age": "s2_1b_age",
}


def _knowledge_neighbours(label, levels):
    """Candidate neighbours of `label` among `levels`, nearest first: the
    kin table for countries, then geography for any place _GEO knows."""
    out = []
    for kin in _KIN.get(str(label), []):
        if kin in levels:
            out.append(kin)
    if str(label) in _GEO:
        here = _GEO[str(label)]
        near = sorted((lv for lv in levels if str(lv) in _GEO and lv != label),
                      key=lambda lv: (_GEO[str(lv)][0] - here[0]) ** 2
                      + (_GEO[str(lv)][1] - here[1]) ** 2)
        out += [lv for lv in near if lv not in out]
    return out


def _geo_columns(records, cats, code):
    """Latitude, longitude per place column, and the distance between the
    first two places found (asylum governorate and origin governorate)."""
    places, names = [], []
    for name in records:
        if records[name]["class"] != "GIVEN":
            continue
        values = cats.get(name)
        if not values:
            continue
        hit = sum(1 for v in values if str(v) in _GEO)
        if hit >= max(4, 0.6 * len(values)):
            table = np.full((len(values) + 1, 2), np.nan, np.float32)
            for idx, v in enumerate(values):
                if str(v) in _GEO:
                    table[idx] = _GEO[str(v)]
            c = code[name].copy()
            c[c < 0] = len(values)
            places.append(table[c])
            names.append(name)
    if not places:
        return None, []
    cols = [p[:, 0] for p in places] + [p[:, 1] for p in places]
    if len(places) >= 2:
        a, b = places[0], places[1]
        la1, lo1, la2, lo2 = [np.radians(x) for x in (a[:, 0], a[:, 1],
                                                       b[:, 0], b[:, 1])]
        h = (np.sin((la2 - la1) / 2) ** 2
             + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
        cols.append((2 * 6371.0 * np.arcsin(np.sqrt(h))).astype(np.float32))
    return np.stack(cols, 1).astype(np.float32), names


def _digest(*arrays):
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# ------------------------------------------------------------- fine-tuning --

def _finetune(path, X, cat_idx, ys, stop_at, seed, say, gated=None):
    """v24: the primary backbone fine-tuned on this instrument's own rows.

    Every item is one meta-dataset -- a context/query split of the rows it is
    given, all items sharing one set of weights -- so what the weights learn
    is how this questionnaire's GIVEN columns carry information, not any one
    item. The caller passes only rows the slice does not hold, so the slice's
    verdict on the result stays honest.

    The schedule is sized to the clock: a few steps are timed, then warmup and
    cosine decay are planned to end by `stop_at` (or at `ft_epochs` passes
    over the items, whichever is sooner). Returns a ClassifierModelSpecs for
    the tuned copy, or None when too few steps fit.
    """
    import copy
    import math
    import torch
    from functools import partial
    from tabpfn import TabPFNClassifier
    from tabpfn.base import ClassifierModelSpecs
    from tabpfn.architectures.interface import PerformanceOptions
    from tabpfn.finetuning._torch_compat import (GradScaler, autocast,
                                                 sdpa_kernel_context)
    from tabpfn.finetuning.data_util import (get_preprocessed_dataset_chunks,
                                            meta_dataset_collator)
    from tabpfn.finetuning.finetuned_base import _ratio_ctx_query_split
    from torch.utils.data import DataLoader

    t0 = time.time()
    rows_min = int(CONFIG.get("ft_rows_min", 64))
    hurdle = CONFIG.get("ft_tasks", "full") == "hurdle"
    data = []

    def add(Xa, ya):
        if len(ya) >= rows_min and np.unique(ya).size >= 2:
            data.append((Xa, ya))

    for j, y in enumerate(ys):
        keep = y >= 0
        w = gated[j] if (hurdle and gated is not None) else None
        if w:
            # the tasks the passes fit: asked-or-not on every row, then the
            # answer on the rows that were asked
            sent = keep & (y == w - 1)
            add(X[keep], sent[keep].astype(np.int64))
            asked = keep & (y < w - 1)
            add(X[asked], y[asked])
        else:
            add(X[keep], y[keep])
    if len(data) < 2:
        return None, {"reason": "items"}
    clf = TabPFNClassifier(
        n_estimators=int(CONFIG.get("ft_estimators", 2)), device="cuda",
        model_path=path, ignore_pretraining_limits=True, random_state=seed,
        categorical_features_indices=list(cat_idx), fit_mode="batched",
        differentiable_input=False,
        inference_config={"ENABLE_GPU_PREPROCESSING": False})
    clf._initialize_model_variables()
    model = clf.models_[0]
    model.to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=float(CONFIG["ft_lr"]),
                            weight_decay=float(CONFIG.get("ft_wd", 0.01)))
    scaler = GradScaler()
    perf = PerformanceOptions(force_recompute_layer=True,
                              use_chunkwise_inference=False)
    max_ctx = int(CONFIG.get("ft_max_ctx", 12000))
    per_pass = sum(max(1, math.ceil(len(yy) / max_ctx)) for _, yy in data)
    cap_steps = int(float(CONFIG.get("ft_epochs", 4)) * per_pass)
    probe = int(CONFIG.get("ft_probe_steps", 4))
    plan = {"total": cap_steps, "warm": max(1, int(0.1 * cap_steps))}

    def lr_at(step):
        if step < plan["warm"]:
            return (step + 1) / plan["warm"]
        frac = (step - plan["warm"]) / max(1, plan["total"] - plan["warm"])
        return 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    step, losses, epoch, planned = 0, [], 0, False
    t_steps = None
    while step < plan["total"]:
        seed_e = int(seed) * 1000 + epoch
        split = partial(_ratio_ctx_query_split,
                        query_ratio=float(CONFIG.get("ft_query_ratio", 0.2)),
                        min_query_size=30, random_state=seed_e)
        ds = get_preprocessed_dataset_chunks(
            calling_instance=clf, X_raw=[a for a, _ in data],
            y_raw=[b for _, b in data], split_fn=split, max_data_size=max_ctx,
            model_type="classifier", equal_split_size=False,
            data_shuffle_seed=seed_e, preprocessing_random_state=int(seed))
        loader = DataLoader(ds, batch_size=1, collate_fn=meta_dataset_collator,
                            shuffle=True,
                            generator=torch.Generator().manual_seed(seed_e))
        for batch in loader:
            if step >= plan["total"] or time.time() > stop_at:
                break
            if step == probe and not planned:
                # the steps so far price the rest of the schedule
                per_step = (time.time() - t_steps) / max(1, probe)
                fit = int((stop_at - time.time()) / max(per_step, 1e-3)) + step
                plan["total"] = min(cap_steps, fit)
                plan["warm"] = max(1, int(0.1 * plan["total"]))
                planned = True
            ctx_u = torch.unique(torch.cat([t.reshape(-1) for t in batch.y_context]))
            qry_u = torch.unique(batch.y_query.reshape(-1))
            if not bool(torch.isin(qry_u, ctx_u).all()):
                continue
            if t_steps is None:
                t_steps = time.time()
            n_cls = int(max(ctx_u.max().item(), qry_u.max().item())) + 1
            clf.n_classes_ = n_cls
            clf.classes_ = np.arange(n_cls)
            opt.zero_grad()
            clf.fit_from_preprocessed(batch.X_context, batch.y_context,
                                      batch.cat_indices, batch.configs,
                                      performance_options=perf)
            with autocast(enabled=True), sdpa_kernel_context():
                logits = clf.forward(batch.X_query, return_raw_logits=True)
                Q, B, E, L = logits.shape
                lg = logits.permute(1, 2, 3, 0).reshape(B * E, L, Q)
                tgt = batch.y_query.repeat(B * E, 1).to("cuda").long()
                loss = torch.nn.functional.cross_entropy(lg.float(), tgt)
            with sdpa_kernel_context():
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            losses.append(float(loss.item()))
        epoch += 1
        if time.time() > stop_at:
            break
    info = {"steps": step, "planned": plan["total"], "cap": cap_steps,
            "items": len(data), "passes": round(step / max(per_pass, 1), 2),
            "loss_first": round(float(np.mean(losses[:20])), 4) if losses else None,
            "loss_last": round(float(np.mean(losses[-20:])), 4) if losses else None,
            "seconds": round(time.time() - t0, 1)}
    del opt, scaler
    if step < int(CONFIG.get("ft_min_steps", 40)) or step < 0.6 * plan["total"]:
        info["reason"] = "too few steps"
        return None, info
    model.eval()
    spec = ClassifierModelSpecs(model=model,
                                architecture_config=copy.deepcopy(clf.configs_[0]),
                                inference_config=copy.deepcopy(clf.inference_config_))
    return spec, info


# --------------------------------------------------------------- calibration --

def _pc(name, phase, default=None):
    """CONFIG[name], or CONFIG[name + "_by_phase"][str(phase)] when a card
    sets the knob per phase (v38: the lat pass is cheap at 900 s and full at
    3,600 s)."""
    by = CONFIG.get(name + "_by_phase") or {}
    if str(phase) in by:
        return by[str(phase)]
    return CONFIG.get(name, default)


def _top_partners(cols, widths):
    """For every column, the other column with the highest mutual
    information on these rows (-1 codes read as one more level); -1 when
    there is no other column."""
    n = len(cols[0])
    offs, blocks = [], []
    o = 0
    for c, w in zip(cols, widths):
        c = np.where(c < 0, w, c).astype(np.int64)
        oh = np.zeros((n, w + 1), np.float32)
        oh[np.arange(n), c] = 1.0
        blocks.append(oh)
        offs.append((o, w + 1))
        o += w + 1
    OH = np.concatenate(blocks, axis=1)
    C = (OH.T @ OH) / float(n)
    marg = np.diag(C).copy()
    J = len(cols)
    best = np.full(J, -1)
    best_mi = np.full(J, -1.0)
    for i in range(J):
        ai, wi = offs[i]
        pi_ = marg[ai:ai + wi]
        for j in range(i + 1, J):
            aj, wj = offs[j]
            P = C[ai:ai + wi, aj:aj + wj]
            E = np.outer(pi_, marg[aj:aj + wj])
            nz = P > 0
            mi = float((P[nz] * np.log(P[nz] / E[nz])).sum())
            if mi > best_mi[i]:
                best_mi[i], best[i] = mi, j
            if mi > best_mi[j]:
                best_mi[j], best[j] = mi, i
    return best


def _latent_types(cols, widths, K, iters, restarts, seed):
    """EM for a K-class latent-class model over categorical columns.

    `cols` are the context rows' codes per item (-1 = missing, read as one
    more level). Returns (hard, loo): the argmax class of every row, and a
    function j -> the argmax class of every row from all items but j. Runs on
    the GPU when there is one; a few seconds for 8,000 rows x 213 items.
    """
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(cols[0])
    offs, blocks = [], []
    o = 0
    for c, w in zip(cols, widths):
        c = np.where(c < 0, w, c).astype(np.int64)
        blocks.append(torch.as_tensor(c + o, device=dev))
        offs.append((o, w + 1))
        o += w + 1
    idx = torch.stack(blocks, 1)                                  # n x J
    OH = torch.zeros((n, o), device=dev, dtype=torch.float32)
    OH.scatter_(1, idx, 1.0)
    alpha = 1.0
    best = None
    for r in range(restarts):
        g = torch.Generator(device=dev).manual_seed(int(seed) * 101 + r)
        R = torch.rand((n, K), generator=g, device=dev) + 0.5
        R = R / R.sum(1, keepdim=True)
        ll_old = -np.inf
        for _ in range(int(iters)):
            pi = (R.sum(0) + alpha) / (n + K * alpha)
            C = R.T @ OH                                          # K x o
            logt = torch.empty_like(C)
            for a, w in offs:
                blk = C[:, a:a + w] + alpha / w
                logt[:, a:a + w] = torch.log(blk / blk.sum(1, keepdim=True))
            L = OH @ logt.T + torch.log(pi)[None, :]
            m = L.max(1, keepdim=True).values
            ll = float((m[:, 0] + torch.log(torch.exp(L - m).sum(1))).sum())
            R = torch.softmax(L, 1)
            if ll - ll_old < 1e-4 * n:
                break
            ll_old = ll
        if best is None or ll > best[0]:
            best = (ll, L.clone(), logt.clone())
    _, L, logt = best
    hard = L.argmax(1).cpu().numpy()

    def loo(j):
        a, w = offs[j]
        own = OH[:, a:a + w] @ logt[:, a:a + w].T
        return (L - own).argmax(1).cpu().numpy()
    return hard, loo


def _compose(parts, width, cal, crowd, floor, big=False):
    """Raw parts -> the vector that leaves predict(), under a calibration.

    cal = (a_sent, a_ans, a_full, a_sent_big). Powers act on the sentinel
    logit, the answer vector and the ungated vector; the fourth replaces the
    first for the items of the largest gate family (UNHCR: the 73 children
    of s3_11), whose sentinel is one hidden attitude answered 73 times and
    may be sharp or blunt in its own way. `floor` mixes the crowd in
    afterwards, exactly as v15 did with its constant 0.004.
    """
    kind = parts[0]
    if kind == "hurdle":
        ps, pa = parts[1], parts[2]
        a_s = cal[3] if (big and len(cal) > 3) else cal[0]
        if a_s != 1.0:
            logit = np.log(ps) - np.log1p(-ps)
            ps = 1.0 / (1.0 + np.exp(-np.clip(a_s * logit, -30.0, 30.0)))
            ps = np.clip(ps, 1e-6, 1.0 - 1e-6)
        if cal[1] != 1.0:
            pa = np.power(np.maximum(pa, 1e-12), cal[1])
            pa = pa / pa.sum(1, keepdims=True)
        P = np.empty((len(ps), width))
        P[:, :width - 1] = pa * (1.0 - ps)[:, None]
        P[:, width - 1] = ps
    elif kind == "full":
        P = parts[1]
        if cal[2] != 1.0:
            P = np.power(np.maximum(P, 1e-12), cal[2])
            P = P / P.sum(1, keepdims=True)
    else:
        P = parts[1]
    P = (1.0 - floor) * P + floor * crowd[None, :]
    if len(cal) > 4 and cal[4] > 0.0:
        P = np.maximum(P - cal[4] * float(CONFIG["scorer_floor"]), 0.0)
    P = np.maximum(P, 1e-9)
    return P / P.sum(1, keepdims=True)


def _slice_loss(results, truth, cal, floor, scorer_floor, big=frozenset()):
    """The grader's mean log score on the slice, sign flipped (lower is
    better), with the grader's own uniform mixing applied."""
    total, count = 0.0, 0
    for name, (parts, width, crowd) in results.items():
        P = _compose(parts, width, cal, crowd, floor, name in big)
        P = scorer_floor + (1.0 - width * scorer_floor) * P
        y = truth[name]
        total -= float(np.log(P[np.arange(len(y)), y]).sum())
        count += len(y)
    return total / max(count, 1)


def _fit_calibration(results, truth, has_hurdle, has_full, big=frozenset()):
    """Coordinate descent over the powers and the floor."""
    powers = list(CONFIG["power_grid"])
    floors = list(CONFIG["floor_grid"])
    cal = [1.0, 1.0, 1.0, 1.0, 0.0]
    floor = float(CONFIG["floor"])
    sf = float(CONFIG["scorer_floor"])
    active = [k for k in range(4)
              if (k in (0, 1) and has_hurdle) or (k == 2 and has_full)
              or (k == 3 and has_hurdle and len(big) > 0)]
    best = _slice_loss(results, truth, tuple(cal), floor, sf, big)
    for _sweep in range(3):
        moved = False
        for k in active:
            for a in powers:
                trial = list(cal)
                trial[k] = a
                if k == 0 and len(big) == 0:
                    trial[3] = a
                loss = _slice_loss(results, truth, tuple(trial), floor, sf, big)
                if loss < best - 1e-12:
                    best, cal, moved = loss, trial, True
        for f in floors:
            loss = _slice_loss(results, truth, tuple(cal), f, sf, big)
            if loss < best - 1e-12:
                best, floor, moved = loss, f, True
        for lam in CONFIG.get("demix_grid", [0.0]):
            trial = list(cal)
            trial[4] = lam
            loss = _slice_loss(results, truth, tuple(trial), floor, sf, big)
            if loss < best - 1e-12:
                best, cal, moved = loss, trial, True
        if not moved:
            break
    return tuple(cal), floor, best


# --------------------------------------------------------------------- main --

def _predict_full(frame, schema):
    records = schema["items"]
    items = [n for n, r in records.items() if r["class"] in ("GIVEN", "PREDICT")]
    given = [n for n in items if records[n]["class"] == "GIVEN"]
    targets = [n for n in items if records[n]["class"] == "PREDICT"]
    sentinel = schema["gated_value"]
    cats, gate, code, isna = {}, {}, {}, {}
    for name in items:
        values = list(records[name]["values"])
        gate[name] = records[name].get("gate")
        cats[name] = values + ([sentinel] if gate[name] else [])
        lookup = {v: i for i, v in enumerate(cats[name])}
        col = frame[name].map(lookup)
        isna[name] = col.isna().to_numpy()
        code[name] = col.fillna(-1).astype(np.int64).to_numpy()
    n = len(frame)
    blank = np.zeros(n, bool)
    for name in targets:
        blank |= isna[name]
    fit_rows, out_rows = np.flatnonzero(~blank), np.flatnonzero(blank)
    key = ("unhcr" if len(targets) >= 150 else "world_bank"
           if len(targets) >= 40 else "unicef" if len(targets) >= 8 else "other")
    _terraform_install()
    _terraform(key)
    stacked = None
    if key in CONFIG["stack_instruments"]:
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        import stack
        _SEEN.add(key)
        stacked = stack.predict(frame, schema)
        if key not in CONFIG["hybrid_instruments"]:
            return stacked
    split = schema.get("split", {})
    phase = 2 if (split.get("n_train") and len(fit_rows)
                  > split["n_train"] + 0.5 * split.get("n_dev", 0)) else 1

    # v22b: an instrument that names TabFM loads it before the clock below is
    # split, so the plan sees what the 6.5 GB read cost instead of finding out
    # inside a gate pass whose estimate never counted it.
    if CONFIG.get("tfm_preload") and key in CONFIG.get("tfm_instruments", []) \
            and "model" not in _TFM and "load_note" not in _TFM:
        _tfm_preload(phase)
    fault = os.environ.get("SB_FAULT", "")
    if fault and fault.split(":")[0] == key and (
            CONFIG.get("gate") or fault.endswith(":all")):
        raise RuntimeError("SB_FAULT injected for %s" % key)

    # ---- the clock: this call's share of what is left of the whole run ----
    budget = CONFIG["budget"][phase] * CONFIG["safety"]
    shares = _pc("share", phase)  # v40: a card may reallocate the 900 s clock only
    share = shares.get(key, 1.0 / 3)
    ahead = sum(v for k, v in shares.items() if k not in _SEEN) or share
    together = max(_STARTED + budget - time.time(), 10.0)
    call_started = time.time()
    deadline = call_started + max(min(together * share / max(ahead, share),
                                      together), 10.0)
    _SEEN.add(key)
    debug = bool(os.environ.get("SB_DEBUG"))

    def say(text):
        if debug:
            print("[debug] %s %s" % (key, text), flush=True)

    # v30: the primary (and the wide) backbone may differ per instrument --
    # TabPFN-3.5 pays only where the GIVEN block is informative and carries a
    # high-cardinality column (UNHCR), and is temperature elsewhere.
    bb_primary = (CONFIG.get("backbone_by_instrument") or {}).get(key) \
        or CONFIG.get("backbone", "v3")
    bb_wide = (CONFIG.get("wide_backbone_by_instrument") or {}).get(key) \
        or CONFIG.get("wide_backbone", "v3")
    path = _checkpoint_name(bb_primary)
    spec = _spec(path) if path else None
    use_tabpfn = bool(path) and _cuda()
    say("backbone %s (wide %s): %s" % (bb_primary, bb_wide,
                                        os.path.basename(path) if path else None))
    if use_tabpfn:
        from tabpfn import TabPFNClassifier
    wide_path = _checkpoint_name(bb_wide) if use_tabpfn else None
    wide_spec = _spec(wide_path) if wide_path else None
    narrow = int(CONFIG.get("narrow_width", 10))
    wide_max = getattr(getattr(wide_spec, "inference_config", None),
                       "MAX_NUMBER_OF_CLASSES", 0) if wide_spec else 0
    wide_ok = wide_spec is not None and wide_max > narrow
    max_width = min(int(CONFIG["max_width"]), wide_max) if wide_ok else narrow
    partner_paths = {}
    ft_holder = {}

    def partner_spec(bb):
        """(checkpoint path, spec) of a second backbone, loaded once."""
        if bb == "ft":
            # the tuned weights live only in memory, once the gate made them
            spec_ = ft_holder.get("spec")
            return ("ft" if spec_ is not None else None), spec_
        if bb not in partner_paths:
            p_ = _checkpoint_name(bb) if use_tabpfn else None
            if bb == "exa":
                partner_paths[bb] = (p_ if p_ and _exa_ok() else None, None)
            elif bb == "kumo":
                partner_paths[bb] = (p_ if p_ and _kumo_ok() else None, None)
                if p_ and _KUMO.get("tuned") and p_ in _KUMO:
                    next(iter(_KUMO[p_].models.values())).load_state_dict(
                        {k: v.to("cuda") for k, v in _KUMO["base_sd:" + p_].items()})
                    _KUMO[p_].eval()
                    _KUMO["tuned"] = False
            elif bb == "tfm":
                partner_paths[bb] = (p_ if p_ and _tfm_ok() else None, None)
            elif bb == "cau":
                partner_paths[bb] = (p_ if p_ and _cau_ok() else None, None)
            else:
                partner_paths[bb] = (p_, _spec(p_) if p_ else None)
        return partner_paths[bb]

    # ---- levels only the held-out rows carry ------------------------------
    # A GIVEN level absent from the visible rows (a country the file's tail
    # holds and its head does not) reaches TabPFN as a missing value; the
    # alternative is to read it as its nearest visible level by the profile
    # of the other GIVEN columns. Which is better is decided on the delivered
    # rows: the largest visible levels are held out in turn, fitted without,
    # and predicted both ways. Nothing changes unless the mapped reading wins
    # by `unseen_margin` skill on those rows.
    unseen_note = {}
    if CONFIG.get("unseen_map") and len(out_rows) and use_tabpfn:
        u_end = time.time() + (deadline - time.time()) * float(
            CONFIG.get("unseen_share", 0.25))
        # the final phase has the clock for a steadier judge
        u_judges = int(CONFIG.get("unseen_judges", 2)) + (1 if phase == 2 else 0)
        u_items = int(CONFIG.get("unseen_items", 40)) * (2 if phase == 2 else 1)
        u_est = int(CONFIG.get("unseen_estimators", 2)) * (2 if phase == 2 else 1)
        for g in given:
            t_u = time.time()
            W = len(cats[g]) + 1
            vis, held = code[g][fit_rows], code[g][out_rows]
            cv = np.bincount(np.where(vis < 0, W - 1, vis), minlength=W)
            ch = np.bincount(np.where(held < 0, W - 1, held), minlength=W)
            unseen = [l for l in range(W - 1)
                      if ch[l] >= int(CONFIG.get("unseen_min_held", 20))
                      and cv[l] < int(CONFIG.get("unseen_max_vis", 5))]
            if not unseen:
                continue
            others = [h for h in given if h != g]
            pool = [v for v in range(W - 1)
                    if cv[v] >= int(CONFIG.get("unseen_pool_min", 50))]
            if len(pool) < 2:
                continue

            def profile(rows, h):
                Wh = len(cats[h]) + 1
                c = code[h][rows]
                pr = np.bincount(np.where(c < 0, Wh - 1, c),
                                 minlength=Wh).astype(float) + 0.5
                return pr / pr.sum()

            def nearest(level_rows, pool_levels, base_rows):
                best, best_d = None, np.inf
                for v in pool_levels:
                    vrows = base_rows[code[g][base_rows] == v]
                    d = 0.0
                    for h in others:
                        a_, b_ = profile(level_rows, h), profile(vrows, h)
                        d += float(np.sqrt(0.5 * np.sum(
                            (np.sqrt(a_) - np.sqrt(b_)) ** 2)))
                    if d < best_d:
                        best, best_d = v, d
                return best, best_d

            def knowledge(level, pool_levels):
                names_ = [cats[g][v] for v in pool_levels]
                for kin in _knowledge_neighbours(cats[g][level], names_):
                    return pool_levels[names_.index(kin)]
                return None

            judges = sorted(pool, key=lambda v: -cv[v])[:u_judges]
            uni = float(np.mean([np.log(len(cats[t])) for t in targets]))
            items_j = list(targets)
            cap_items = u_items
            if len(items_j) > cap_items:
                step = len(items_j) / cap_items
                items_j = [items_j[int(i * step)] for i in range(cap_items)]
            Xb = np.zeros((n, len(given)), np.float32)
            for j, name in enumerate(given):
                c = code[name].astype(np.float32)
                c[c < 0] = len(cats[name])
                Xb[:, j] = c
            gi = given.index(g)
            sf = float(CONFIG["scorer_floor"])
            urng = np.random.default_rng(int(CONFIG["seed"]) + 5)
            gains, gains_kin, cells, ok = 0.0, 0.0, 0, True
            for L in judges:
                if time.time() > u_end:
                    ok = False
                    break
                q = fit_rows[code[g][fit_rows] == L]
                qcap = int(CONFIG.get("unseen_query_cap", 400))
                if len(q) > qcap:
                    q = np.sort(urng.choice(q, qcap, replace=False))
                tr_ = fit_rows[code[g][fit_rows] != L]
                if 0 < CONFIG["rows_cap"] < len(tr_):
                    tr_ = np.sort(urng.choice(tr_, CONFIG["rows_cap"],
                                              replace=False))
                others_pool = [v for v in pool if v != L]
                nb, _ = nearest(q, others_pool, fit_rows)
                kb = knowledge(L, others_pool)
                Xq_as = Xb[q]
                Xq_map = Xb[q].copy()
                Xq_map[:, gi] = float(nb)
                Xq_kin = None
                if kb is not None and kb != nb:
                    Xq_kin = Xb[q].copy()
                    Xq_kin[:, gi] = float(kb)
                for name in items_j:
                    if time.time() > u_end:
                        ok = False
                        break
                    y = code[name][tr_]
                    keep = y >= 0
                    yq = code[name][q]
                    kq = yq >= 0
                    width = len(cats[name])
                    if keep.sum() < 64 or kq.sum() < 20 \
                            or np.unique(y[keep]).size < 2:
                        continue
                    try:
                        m_ = TabPFNClassifier(
                            n_estimators=u_est,
                            device="cuda",
                            model_path=spec if spec is not None else path,
                            ignore_pretraining_limits=True,
                            random_state=int(CONFIG["seed"]),
                            categorical_features_indices=list(range(len(given))))
                        m_.fit(Xb[tr_][keep], y[keep])
                        cls = np.asarray(m_.classes_, np.int64)
                        P1 = np.zeros((len(q), width))
                        P1[:, cls] = m_.predict_proba(Xq_as)
                        P2 = np.zeros((len(q), width))
                        P2[:, cls] = m_.predict_proba(Xq_map)
                        P3 = None
                        if Xq_kin is not None:
                            P3 = np.zeros((len(q), width))
                            P3[:, cls] = m_.predict_proba(Xq_kin)
                    except Exception:
                        continue
                    rows_ = np.flatnonzero(kq)

                    def _ll(P):
                        P = (P + 1e-9) / (P + 1e-9).sum(1, keepdims=True)
                        P = sf + (1 - width * sf) * P
                        return float(np.log(P[rows_, yq[rows_]]).sum())
                    l1 = _ll(P1)
                    gains += _ll(P2) - l1
                    gains_kin += (_ll(P3) - l1) if P3 is not None else (
                        _ll(P2) - l1)
                    cells += len(rows_)
                if not ok:
                    break
            gain = gains / cells / uni if cells else 0.0
            gain_kin = gains_kin / cells / uni if cells else 0.0
            use_kin = gain_kin > gain
            best_gain = max(gain, gain_kin)
            apply = ok and cells > 0 and best_gain > float(
                CONFIG.get("unseen_margin", 0.002))
            mapping = {}
            if apply:
                for l in unseen:
                    lrows = out_rows[code[g][out_rows] == l]
                    nb, _ = nearest(lrows, pool, fit_rows)
                    if use_kin:
                        kb = knowledge(l, pool)
                        nb = kb if kb is not None else nb
                    mapping[l] = nb
                    code[g][lrows] = nb
            unseen_note[g] = {
                "unseen": [cats[g][l] for l in unseen],
                "held_rows": int(sum(ch[l] for l in unseen)),
                "judge_gain": round(gain, 4), "judge_gain_kin": round(gain_kin, 4),
                "by": "knowledge" if (apply and use_kin) else ("profile" if apply else None),
                "cells": int(cells),
                "applied": bool(apply), "complete": bool(ok),
                "map": {cats[g][l]: cats[g][v] for l, v in mapping.items()},
                "seconds": round(time.time() - t_u, 1)}
            say("unseen levels in %s: %s" % (g, unseen_note[g]))

    # ---- features: raw integer codes, unseen level as its own -------------
    X = np.zeros((n, len(given)), np.float32)
    for j, name in enumerate(given):
        c = code[name].astype(np.float32)
        c[c < 0] = len(cats[name])
        X[:, j] = c
    categorical = list(range(len(given)))

    # The household total: the one column attention over eight counts does
    # not get for free. Built once, used only by the `roster` candidate.
    rosters = {}
    for name in given:
        scale = _ordered(records[name]["values"])
        if scale is None:
            continue
        seen = [v for v in scale if v is not None]
        if min(seen) != 0.0:
            continue
        table = np.full(len(cats[name]) + 1, np.nan, np.float32)
        for idx, value in enumerate(scale):
            if value is not None and idx < len(cats[name]):
                table[idx] = value
        c = code[name].copy()
        c[c < 0] = len(cats[name])
        rosters.setdefault("_".join(name.split("_")[:2]), []).append(table[c])
    roster_cols = [np.nansum(np.stack(cols), axis=0)
                   for _, cols in sorted(rosters.items()) if len(cols) >= 3]
    X_roster = (np.concatenate([X, np.stack(roster_cols, 1).astype(np.float32)],
                               axis=1) if roster_cols else None)
    # Places: coordinates and the distance home, numeric columns after the codes.
    geo_cols, geo_names = _geo_columns(records, cats, code)
    X_geo, cat_geo = None, None
    if geo_cols is not None:
        drop = set()
        if CONFIG.get("geo_replace", True):
            drop = {given.index(nm) for nm in geo_names
                    if nm in given and len(cats[nm]) > 30}
        keep_j = [j for j in range(len(given)) if j not in drop]
        X_geo = np.concatenate([X[:, keep_j], geo_cols], axis=1).astype(
            np.float32)
        cat_geo = list(range(len(keep_j)))
    # Ordered scales as numbers: the columns whose labels carry a real scale
    # (age bands, wage bands, roster counts; not a plain 1..K numbering)
    # leave the categorical list and are read as their numeric value, so
    # every ensemble member sees the order instead of half of them.
    X_ord, cat_ord, ord_cols = X.copy(), list(categorical), []
    for j, name in enumerate(given):
        scale = _ordered(records[name]["values"])
        if scale is None:
            continue
        seen = [v for v in scale if v is not None]
        if seen == list(range(int(seen[0]), int(seen[0]) + len(seen))):
            continue
        table = np.full(len(cats[name]) + 1, np.nan, np.float32)
        for idx, value in enumerate(scale):
            if value is not None and idx < len(cats[name]):
                table[idx] = value
        c = code[name].copy()
        c[c < 0] = len(cats[name])
        X_ord[:, j] = table[c]
        cat_ord.remove(j)
        ord_cols.append(name)
    if ord_cols:
        # two columns on the same scale (the two wage bands) also enter as
        # their difference, the one number attention does not form for free
        extra = []
        for a_i in range(len(ord_cols)):
            for b_i in range(a_i + 1, len(ord_cols)):
                a_, b_ = ord_cols[a_i], ord_cols[b_i]
                if list(records[a_]["values"]) == list(records[b_]["values"]):
                    ja, jb = given.index(a_), given.index(b_)
                    extra.append((X_ord[:, ja] - X_ord[:, jb]).astype(np.float32))
        if extra:
            X_ord = np.concatenate([X_ord, np.stack(extra, 1)], axis=1)
    if not ord_cols:
        X_ord = None
    # Non-response levels per item (never the sentinel), for the nested hurdle.
    spec_idx = {}
    if CONFIG.get("nest"):
        _rx = re.compile(str(CONFIG.get("nest_pattern", "")), re.I)
        for name in targets:
            vals = list(records[name]["values"])
            sp = [i for i, v in enumerate(vals) if _rx.search(str(v))]
            if sp and len(sp) < len(vals):
                spec_idx[name] = np.asarray(sp, np.int64)
    # Families of items that share (nearly) one option list, for the pooled
    # fits: labels are canonicalised and a list joins a family when it
    # overlaps the family's union enough. Each member maps its own codes
    # into the union and reads its own labels back out of it.
    pool_family, pool_members, pool_maps = {}, {}, {}
    if CONFIG.get("pool"):
        def _canon(v):
            return re.sub(r"[^a-z0-9]+", "", str(v).lower())
        fams = []          # [union_labels(list), members(list)]
        for name in targets:
            vals = [_canon(v) for v in records[name]["values"]]
            if len(vals) < int(CONFIG.get("pool_min_width", 6)):
                continue
            sv = set(vals)
            best, best_j = None, 0.0
            for fam in fams:
                su = set(fam[0])
                jac = len(sv & su) / max(len(sv | su), 1)
                if jac > best_j:
                    best, best_j = fam, jac
            if best is not None and best_j >= float(CONFIG.get("pool_jaccard", 0.75)):
                for v in vals:
                    if v not in best[0]:
                        best[0].append(v)
                best[1].append(name)
            else:
                fams.append([list(dict.fromkeys(vals)), [name]])
        for union, members in fams:
            if 2 <= len(members) <= int(CONFIG.get("pool_max_members", 8)):
                fid = len(pool_members)
                pool_members[fid] = list(members)
                uidx = {u: i for i, u in enumerate(union)}
                for m in members:
                    pool_family[m] = fid
                    vals = [_canon(v) for v in records[m]["values"]]
                    to_u = np.asarray([uidx[v] for v in vals], np.int64)
                    pool_maps[m] = (to_u, len(union))
    # Per-item column selection by bias-corrected mutual information.
    select_map = None
    if CONFIG.get("select_frac", 0) > 0:
        def _mi_mm(x, y, nx, ny):
            joint = np.zeros((nx, ny))
            np.add.at(joint, (x, y), 1.0)
            tot = joint.sum()
            if tot <= 0:
                return 0.0
            joint /= tot
            px = joint.sum(1, keepdims=True)
            py = joint.sum(0, keepdims=True)
            nz = joint > 0
            mi = float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())
            kx, ky = int((px > 0).sum()), int((py > 0).sum())
            return mi - (int(nz.sum()) - kx - ky + 1) / (2.0 * tot)
        select_map, dropped = {}, 0
        for name in targets:
            y = code[name][fit_rows]
            keep = y >= 0
            if keep.sum() < 64:
                continue
            width = len(cats[name])
            scores = np.zeros(len(given))
            for j, g in enumerate(given):
                x = X[fit_rows, j].astype(np.int64)
                scores[j] = _mi_mm(x[keep], y[keep], len(cats[g]) + 1, width)
            top = float(scores.max())
            cols = [j for j in range(len(given))
                    if scores[j] >= float(CONFIG["select_frac"]) * max(top, 1e-9)
                    and scores[j] > 0]
            if len(cols) < int(CONFIG.get("select_min", 2)):
                cols = list(np.argsort(-scores)[:int(CONFIG.get("select_min", 2))])
            cols = sorted(int(c) for c in cols)
            if len(cols) < len(given):
                dropped += 1
            select_map[name] = np.asarray(cols, np.int64)
        if not dropped:
            select_map = None
    # One-hot codes as numeric columns: a view where no ensemble member
    # reads a category list as an ordinal scale.
    X_onehot, cat_onehot = None, []
    hi_card = [j for j, name in enumerate(given)
               if len(cats[name]) > int(CONFIG.get("onehot_hi_card", 30))]
    n_onehot = sum(len(cats[name]) + 1 for name in given)
    if hi_card:
        # v25: TabPFN reads a column with more than 30 levels as a number
        # (alphabetical governorates); only those columns are one-hot, the
        # rest stay categorical codes, so the pass costs a fraction of a
        # full one-hot and the slice judges the place columns alone.
        low = [j for j in range(len(given)) if j not in hi_card]
        n_hi = sum(len(cats[given[j]]) + 1 for j in hi_card)
        X_onehot = np.zeros((n, len(low) + n_hi), np.float32)
        X_onehot[:, :len(low)] = X[:, low]
        cat_onehot = list(range(len(low)))
        off = len(low)
        for j in hi_card:
            X_onehot[np.arange(n), off + X[:, j].astype(np.int64)] = 1.0
            off += len(cats[given[j]]) + 1
    elif n_onehot <= int(CONFIG.get("onehot_max", 0)):
        X_onehot = np.zeros((n, n_onehot), np.float32)
        off = 0
        for j, name in enumerate(given):
            X_onehot[np.arange(n), off + X[:, j].astype(np.int64)] = 1.0
            off += len(cats[name]) + 1

    rng = np.random.default_rng(CONFIG["seed"])
    take_all = fit_rows
    if 0 < CONFIG["rows_cap"] < len(take_all):
        take_all = take_all[rng.choice(len(take_all), CONFIG["rows_cap"],
                                       replace=False)]
    take_all = np.sort(take_all)

    # ---- v25: parents with PREDICT children, and the split column ---------
    parent_children = {}
    for name in targets:
        g_ = gate[name]
        if g_ and g_["parent"] in cats and records[g_["parent"]]["class"] == "PREDICT":
            parent_children.setdefault(g_["parent"], []).append(name)
    chain_parents = [p_ for p_, ch_ in sorted(parent_children.items(),
                                              key=lambda kv: -len(kv[1]))
                     if len(ch_) >= int(CONFIG.get("chain_min_children", 3))]
    chain_parents = chain_parents[:int(CONFIG.get("chain_max_parents", 6))]
    if not CONFIG.get("chain"):
        chain_parents = []
    if not CONFIG.get("prop"):
        parent_children = {}
    split_col = None
    if CONFIG.get("split") and use_tabpfn and len(take_all):
        best_s, best_mi = None, -1.0
        prefer_ = [g_ for g_ in given if re.search(
            str(CONFIG.get("split_prefer", "country")), g_, re.I)]
        for j_, g_ in enumerate(given):
            if g_ == "wave":
                continue          # the round is handled by recode / wave_fit
            if prefer_ and g_ not in prefer_:
                continue          # the survey country, when there is one
            x_ = X[take_all, j_].astype(np.int64)
            lv_, cnt_ = np.unique(x_, return_counts=True)
            if int((cnt_ >= int(CONFIG.get("split_min_rows", 300))).sum()) < 2:
                continue
            tot_ = 0.0
            for name in targets:
                y_ = code[name][take_all]
                k_ = y_ >= 0
                if k_.sum() < 64:
                    continue
                tot_ += _mi(x_[k_], y_[k_], len(cats[g_]) + 1, len(cats[name]))
            if tot_ > best_mi:
                best_s, best_mi = j_, tot_
        split_col = best_s


    # ---- gates the frame already answers --------------------------------
    closed, checked, opens_of = {}, [], {}
    for name in targets:
        g = gate[name]
        if not g or g["parent"] not in cats:
            continue
        parent = g["parent"]
        want = set(g["observed_if"])
        opens = np.zeros(len(cats[parent]) + 1, bool)
        for idx, value in enumerate(cats[parent]):
            opens[idx] = value in want
        opens_of[name] = opens
        if records[parent]["class"] == "GIVEN":
            pc = code[parent].copy()
            pc[pc < 0] = len(cats[parent])
            shut = ~opens[pc]
            truth = code[name][fit_rows] == len(cats[name]) - 1
            seen = code[name][fit_rows] >= 0
            agree = bool(seen.any()) and bool(
                np.array_equal(shut[fit_rows][seen], truth[seen]))
            checked.append((name, agree))
            if agree:
                closed[name] = shut

    # Siblings for the recode view: items of one multi-select battery (same
    # name stem, same gate parent). A round-1 row is rewritten as a skip only
    # when the parent routes away AND no sibling was ticked -- a respondent who
    # answered the battery despite the parent's value keeps every answer.
    ms_siblings, zero_code = {}, {}
    for name in targets:
        if not gate[name] or name not in opens_of:
            continue
        stem = "_".join(name.split("_")[:2])
        sibs = [t for t in targets if t != name and gate[t]
                and gate[t]["parent"] == gate[name]["parent"]
                and "_".join(t.split("_")[:2]) == stem]
        ms_siblings[name] = sibs
        vals = list(records[name]["values"])
        zero_code[name] = next((i for i, v in enumerate(vals)
                                if str(v).strip().split(".")[0].strip() == "0"), 0)
    # Twins for the delta view: item code -> union index, GIVEN code -> union
    # index (-1 where the GIVEN level has no counterpart), and the GIVEN column.
    twin_map = {}
    if CONFIG.get("delta"):
        def _cn(v):
            return re.sub(r"[^a-z0-9]+", "", str(v).lower())
        for name, gname in _TWINS.items():
            if name not in targets or gname not in given:
                continue
            iv = [_cn(v) for v in records[name]["values"]]
            gv = [_cn(v) for v in records[gname]["values"]]
            if len(set(iv) & set(gv)) / max(len(set(iv) | set(gv)), 1) < 0.75:
                continue
            union = list(dict.fromkeys(gv + iv))
            uidx = {u: i for i, u in enumerate(union)}
            to_u_item = np.asarray([uidx[v] for v in iv], np.int64)
            to_u_given = np.asarray([uidx[v] for v in gv] + [-1], np.int64)
            twin_map[name] = (given.index(gname), to_u_item, to_u_given, len(union))

    # ---- one round, or two? ---------------------------------------------
    wave_mode = str(CONFIG.get("wave_fit", "off"))
    wave_col = "wave" if "wave" in given else None
    wave_star = None
    if wave_col and wave_mode != "off" and len(out_rows):
        held = code[wave_col][out_rows]
        levels = np.unique(held[held >= 0])
        if levels.size == 1:
            wave_star = int(levels[0])
            if int((code[wave_col][take_all] == wave_star).sum()) \
                    < int(CONFIG["wave_floor"]):
                wave_star = None

    # ---- one pass: fit on `take`, predict `query` ------------------------
    cache = {}

    def fit_predict(Xa, ya, Xq, width, estimators, backbone, tag,
                    cat_idx=None, seed=None):
        """One TabPFN fit. Identical (rows, labels) fits are served once."""
        cat_idx = categorical if cat_idx is None else cat_idx
        seed = int(CONFIG["seed"]) if seed is None else int(seed)
        key_ = (tag, backbone, width, int(estimators), len(ya),
                _digest(ya), _digest(Xa), tuple(cat_idx), seed)
        if key_ in cache:
            return cache[key_]
        if backbone == "tfm":
            use_path, _unused = partner_spec("tfm")
            if use_path is None:
                raise ValueError("no tfm backbone")
            if width > int(CONFIG.get("tfm_max_width", 10)) or \
                    np.unique(ya).size > 10:
                raise ValueError("tfm cannot carry %d classes" % width)
            model = _tfm_fit(Xa, ya, int(CONFIG.get("tfm_estimators", 2)),
                             seed)
            full = np.zeros((len(Xq), width))
            full[:, np.asarray(model.classes_, np.int64)] = \
                model.predict_proba(np.asarray(Xq, np.float32))
            cache[key_] = full
            return full
        if backbone == "exa":
            use_path, _unused = partner_spec("exa")
            if use_path is None:
                raise ValueError("no exa backbone")
            if width > int(CONFIG.get("exa_max_width", narrow)):
                # the item keeps the primary vector (combine drops the pass)
                raise ValueError("exa cannot carry %d classes" % width)
            model = _exa_fit(use_path, Xa, ya, estimators, seed)
            full = np.zeros((len(Xq), width))
            full[:, np.asarray(model.classes_, np.int64)] = \
                model.predict_proba(np.asarray(Xq, np.float32))
            cache[key_] = full
            return full
        if backbone == "kumo":
            use_path, _unused = partner_spec("kumo")
            if use_path is None:
                raise ValueError("no kumo backbone")
            if width > int(CONFIG.get("kumo_max_width", narrow)):
                raise ValueError("kumo cannot carry %d classes" % width)
            model = _kumo_fit(use_path, Xa, ya, estimators, seed)
            full = np.zeros((len(Xq), width))
            full[:, np.asarray(model.classes_, np.int64)] = \
                model.predict_proba(np.asarray(Xq, np.float32))
            cache[key_] = full
            return full
        if backbone == "cau":
            use_path, _unused = partner_spec("cau")
            if use_path is None:
                raise ValueError("no cau backbone")
            if width > int(CONFIG.get("cau_max_width", narrow)) or \
                    np.unique(ya).size > 10:
                raise ValueError("cau cannot carry %d classes" % width)
            model = _cau_fit(Xa, ya, estimators, seed)
            full = np.zeros((len(Xq), width))
            full[:, np.asarray(model.classes_, np.int64)] = \
                model.predict_proba(np.asarray(Xq, np.float32))
            cache[key_] = full
            return full
        bb_bin = CONFIG.get("binary_backbone")
        if width == 2 and bb_bin and backbone == "v3":      # "v3" = the primary
            use_path, use_spec = partner_spec(bb_bin)
            if use_path is None:
                raise ValueError("no %s backbone" % bb_bin)
        elif backbone != "v3":
            use_path, use_spec = partner_spec(backbone)
            if use_path is None:
                raise ValueError("no %s backbone" % backbone)
            limit = getattr(getattr(use_spec, "inference_config", None),
                            "MAX_NUMBER_OF_CLASSES", narrow) if use_spec \
                else narrow
            if width > limit:
                # the item keeps the primary vector (combine drops the pass)
                raise ValueError("%s cannot carry %d classes"
                                 % (backbone, width))
        elif width > narrow and wide_ok:
            use_spec, use_path = wide_spec, wide_path
        else:
            use_spec, use_path = spec, path
        model = TabPFNClassifier(
            n_estimators=int(estimators), device="cuda",
            model_path=use_spec if use_spec is not None else use_path,
            ignore_pretraining_limits=True, random_state=seed,
            categorical_features_indices=list(cat_idx),
            inference_config=_inference_override(key),
            **_clf_kwargs(key))
        model.fit(Xa, ya)
        pr = model.predict_proba(Xq)
        full = np.zeros((len(Xq), width))
        full[:, np.asarray(model.classes_, np.int64)] = pr
        cache[key_] = full
        return full

    pool_cache = {}

    def fit_model(Xa, ya, estimators, backbone, cat_idx, seed):
        """A fitted TabPFN (primary or partner checkpoint), for callers that
        query it more than once."""
        if backbone == "tfm":
            use_path, _unused = partner_spec("tfm")
            if use_path is None or np.unique(ya).size > 10:
                raise ValueError("no tfm backbone for this fit")
            return _tfm_fit(Xa, ya, int(CONFIG.get("tfm_estimators", 2)), seed)
        if backbone == "exa":
            use_path, _unused = partner_spec("exa")
            if use_path is None:
                raise ValueError("no exa backbone")
            return _exa_fit(use_path, Xa, ya, estimators, seed)
        if backbone == "kumo":
            use_path, _unused = partner_spec("kumo")
            if use_path is None:
                raise ValueError("no kumo backbone")
            return _kumo_fit(use_path, Xa, ya, estimators, seed)
        if backbone == "cau":
            use_path, _unused = partner_spec("cau")
            if use_path is None or np.unique(ya).size > 10:
                raise ValueError("no cau backbone for this fit")
            return _cau_fit(Xa, ya, estimators, seed)
        if backbone != "v3":
            use_path, use_spec = partner_spec(backbone)
            if use_path is None:
                raise ValueError("no %s backbone" % backbone)
        else:
            use_spec, use_path = spec, path
        model = TabPFNClassifier(
            n_estimators=int(estimators), device="cuda",
            model_path=use_spec if use_spec is not None else use_path,
            ignore_pretraining_limits=True, random_state=int(seed),
            categorical_features_indices=list(cat_idx),
            inference_config=_inference_override(key),
            **_clf_kwargs(key))
        model.fit(Xa, ya)
        return model

    def table_member(name, width, take, query):
        y = code[name][take]
        keep = y >= 0
        best, best_mi = None, -1.0
        for g in given:
            x = code[g][take]
            xx = np.where(x < 0, len(cats[g]), x)
            mi = _mi(xx[keep], y[keep], len(cats[g]) + 1, width)
            if mi > best_mi:
                best, best_mi = g, mi
        x = code[best]
        xx = np.where(x < 0, len(cats[best]), x)
        table = np.full((len(cats[best]) + 1, width), CONFIG["smoothing"])
        np.add.at(table, (xx[take][keep], y[keep]), 1.0)
        crowd = table.sum(0) / table.sum()
        table += 4.0 * crowd[None, :]
        table /= table.sum(1, keepdims=True)
        return table[xx[query]]

    def run_pass(take, query, estimators, est_sent, backbone="v3",
                 wave_all=False, dup_wave=False, features=None, tag="",
                 stop_at=None, cat_idx=None, only=None, seed=None,
                 select=None, nest=False, pool=False, recode=False,
                 delta=False, prop=False, chain=False, split=False,
                 lat=False):
        """Every item once: raw parts per item, plus the crowd marginal.

        Returns {name: (parts, width, crowd)} where parts is
        ("hurdle", ps, pa) | ("full", P) | ("crowd", P). `stop_at` is the
        moment after which no further TabPFN fit is started -- items past
        it fall to the crowd, which is what v15 did at its deadline.
        """
        Xf = X if features is None else features
        stop_at = deadline if stop_at is None else stop_at
        wave_ok = None
        if wave_star is not None:
            wave_ok = code[wave_col][take] == wave_star
            if int(wave_ok.sum()) < int(CONFIG["wave_floor"]):
                wave_ok = None
        # The scored round twice in the context: a weight of two on the
        # rows that are the population, without the loss of the other.
        if dup_wave and wave_ok is not None:
            take = np.concatenate([take, take[wave_ok]])
            wave_ok = np.concatenate([wave_ok, np.ones(int(wave_ok.sum()), bool)])
        Xtr_all, Xq_all = Xf[take], Xf[query]
        cat_all = categorical if cat_idx is None else list(cat_idx)
        # v25 chain: the largest PREDICT parents, predicted out of fold on
        # the context rows and in full on the query rows, as numeric columns
        if chain and chain_parents:
            folds_ = int(CONFIG.get("chain_folds", 2))
            crng_ = np.random.default_rng(int(CONFIG["seed"]) + 13)
            fold_id = crng_.integers(0, folds_, size=len(take))
            oof_cols, q_cols = [], []
            for par in chain_parents:
                wp = len(cats[par])
                yp = code[par][take]
                kp = yp >= 0
                if kp.sum() < 64 or np.unique(yp[kp]).size < 2:
                    continue
                cp = np.bincount(yp[kp], minlength=wp).astype(float) + 0.5
                O = np.tile(cp / cp.sum(), (len(take), 1))
                try:
                    for f_ in range(folds_):
                        tr_ = kp & (fold_id != f_)
                        te_ = fold_id == f_
                        if tr_.sum() < 64 or np.unique(yp[tr_]).size < 2 \
                                or not te_.any():
                            continue
                        P_ = fit_predict(Xtr_all[tr_], yp[tr_], Xtr_all[te_], wp,
                                         estimators, backbone,
                                         tag + "_chain%d" % f_, cat_all, seed)
                        O[te_] = P_ / np.maximum(P_.sum(1, keepdims=True), 1e-12)
                    Q = fit_predict(Xtr_all[kp], yp[kp], Xq_all, wp, estimators,
                                    backbone, tag, cat_all, seed)
                    Q = Q / np.maximum(Q.sum(1, keepdims=True), 1e-12)
                except Exception:
                    continue
                oof_cols.append(O.astype(np.float32))
                q_cols.append(Q.astype(np.float32))
            if oof_cols:
                Xtr_all = np.concatenate([Xtr_all] + oof_cols, axis=1)
                Xq_all = np.concatenate([Xq_all] + q_cols, axis=1)
        # v38 lat: the latent type of every context row, from the answer
        # matrix, and p(type | x) on the query rows (one fit). `lat_ks`
        # lists several decompositions whose integrated vectors are averaged.
        lat_on = False
        lat_dec = []
        lp_mode0 = (_pc("lat_parent_by_instrument", phase) or {}).get(key, CONFIG.get("lat_parent"))
        lp_only = lp_mode0 in ("only", "partner")
        lat_partner = {}
        if lat and lp_mode0 == "partner" and select is None and use_tabpfn:
            try:
                bp_ = _top_partners([code[n_][take] for n_ in targets],
                                    [len(cats[n_]) for n_ in targets])
                lat_partner = {n_: targets[int(b_)] for n_, b_ in zip(targets, bp_)
                               if int(b_) >= 0}
            except Exception:
                lat_partner = {}
        if lat and select is None and use_tabpfn and len(targets) >= int(
                CONFIG.get("lat_min_items", 8)):
            for Kl in ([] if lp_only else
                       (_pc("lat_ks", phase) or [int(_pc("lat_k", phase, 8))])):
                try:
                    Kl = int(Kl)
                    lat_hard, lat_loo = _latent_types(
                        [code[n_][take] for n_ in targets],
                        [len(cats[n_]) for n_ in targets], Kl,
                        CONFIG.get("lat_iters", 120), CONFIG.get("lat_restarts", 2),
                        int(CONFIG["seed"]) if seed is None else int(seed))
                    if np.unique(lat_hard).size < 2:
                        continue
                    Pk = fit_predict(Xtr_all, lat_hard, Xq_all, Kl, estimators,
                                     backbone, tag + "_latk%d" % Kl, cat_all, seed)
                    Pk = np.maximum(Pk, 1e-9)
                    Pk = Pk / Pk.sum(1, keepdims=True)
                    Xq_lat = np.concatenate(
                        [np.repeat(Xq_all, Kl, axis=0),
                         np.tile(np.arange(Kl, dtype=np.float32),
                                 len(Xq_all))[:, None]], axis=1)
                    lat_dec.append((Kl, lat_loo, Pk, Xq_lat))
                except Exception:
                    continue
            if lat_dec or lp_only:
                cat_lat = list(cat_all) + [Xtr_all.shape[1]]
                lat_index = {n_: i_ for i_, n_ in enumerate(targets)}
                lat_on = True
        par_cache = {}

        def parent_vector(par, Xtr_p, Xq_p, cat_p):
            """P(parent level | x) on the query rows, len(cats[par]) columns
            (the sentinel last when the parent is gated); None on failure."""
            if par in par_cache:
                return par_cache[par]
            wp = len(cats[par])
            yp = code[par][take]
            kp = yp >= 0
            P = None
            # v43: six parents route 156 children, so their own fit may run at a
            # larger E than the pass (off unless the card names it)
            pe_ = int(_pc("lat_parent_estimators", phase) or 0)
            e_par = max(estimators, pe_) if pe_ else estimators
            s_par = max(est_sent, pe_) if pe_ else est_sent
            try:
                if kp.sum() < 64 or np.unique(yp[kp]).size < 2:
                    c_ = np.bincount(yp[kp], minlength=wp).astype(float) + 0.5
                    P = np.tile(c_ / c_.sum(), (len(Xq_p), 1))
                elif gate[par]:
                    is_s = (yp == wp - 1).astype(np.int64)
                    if is_s[kp].min() == is_s[kp].max():
                        ps_ = np.full(len(Xq_p), float(is_s[kp].mean()))
                    else:
                        ps_ = fit_predict(Xtr_p[kp], is_s[kp], Xq_p, 2, s_par,
                                          backbone, tag, cat_p, seed)[:, 1]
                    ak = kp & (yp < wp - 1)
                    if ak.sum() >= 32 and np.unique(yp[ak]).size >= 2:
                        pa_ = fit_predict(Xtr_p[ak], yp[ak], Xq_p, wp - 1,
                                          e_par, backbone, tag, cat_p, seed)
                    else:
                        c_ = np.bincount(yp[ak], minlength=wp - 1).astype(float) + 0.5
                        pa_ = np.tile(c_ / c_.sum(), (len(Xq_p), 1))
                    pa_ = pa_ / np.maximum(pa_.sum(1, keepdims=True), 1e-12)
                    ps_ = np.clip(ps_, 1e-6, 1 - 1e-6)
                    P = np.empty((len(Xq_p), wp))
                    P[:, :wp - 1] = pa_ * (1.0 - ps_)[:, None]
                    P[:, wp - 1] = ps_
                else:
                    P = fit_predict(Xtr_p[kp], yp[kp], Xq_p, wp, e_par,
                                    backbone, tag, cat_p, seed)
                    P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
            except Exception:
                P = None
            par_cache[par] = P
            return P

        def fp(Xa_, ya_, Xq_, width_, est_, bb_, tag_, cat_, seed_):
            """fit_predict, or -- in the split view -- one fit per level of
            the split column with the instrument-wide fit for the rest."""
            if not split or split_col is None or Xa_.shape[1] <= split_col:
                return fit_predict(Xa_, ya_, Xq_, width_, est_, bb_, tag_, cat_, seed_)
            full_ = np.zeros((len(Xq_), width_))
            done_ = np.zeros(len(Xq_), bool)
            for lv_ in np.unique(Xq_[:, split_col]):
                ma_ = Xa_[:, split_col] == lv_
                mq_ = Xq_[:, split_col] == lv_
                if int(ma_.sum()) >= int(CONFIG.get("split_min_rows", 300)) \
                        and np.unique(ya_[ma_]).size >= 2:
                    try:
                        P_ = fit_predict(Xa_[ma_], ya_[ma_], Xq_[mq_], width_, est_,
                                         bb_, tag_ + "_split", cat_, seed_)
                    except Exception:
                        continue
                    full_[mq_] = P_
                    done_[mq_] = True
            if not done_.all():
                P_ = fit_predict(Xa_, ya_, Xq_, width_, est_, bb_, tag_, cat_, seed_)
                full_[~done_] = P_[~done_]
            return full_
        out = {}
        fits, cut = 0, 0
        for name in targets:
            width = len(cats[name])
            Xtr, Xq, cat_i = Xtr_all, Xq_all, cat_all
            if select is not None and name in select:
                cols = select[name]
                Xtr, Xq = Xtr_all[:, cols], Xq_all[:, cols]
                cset = set(cat_all)
                cat_i = [i for i, c in enumerate(cols) if int(c) in cset]
            y_all = code[name][take]
            keep = y_all >= 0
            gated = bool(gate[name])
            zero_q = None          # query rows whose round wrote 0 for a skip
            if recode and gated and wave_col is not None and name in opens_of \
                    and records[gate[name]["parent"]]["class"] == "PREDICT":
                wv = code[wave_col][take]
                sent = y_all == width - 1
                levels_w = [w for w in np.unique(wv[wv >= 0])
                            if int(((wv == w) & keep).sum()) >= CONFIG["wave_floor"]]
                with_s = [w for w in levels_w if bool(sent[(wv == w) & keep].any())]
                without = [w for w in levels_w if w not in with_s]
                if with_s and without:
                    par = gate[name]["parent"]
                    pc = code[par][take]
                    pc_full = np.where(pc < 0, len(cats[par]), pc)
                    shut = ~opens_of[name][pc_full]
                    zero_rows = np.isin(wv, without) & keep & (pc >= 0)
                    untouched = y_all == zero_code.get(name, 0)
                    for sib in ms_siblings.get(name, []):
                        cs = code[sib][take]
                        untouched &= (cs < 0) | (cs == zero_code.get(sib, 0))
                    y_all = y_all.copy()
                    if recode == "emp":
                        # v20: the routing the rounds with a sentinel actually
                        # followed, per parent level, instead of the declared
                        # gate. Agreement between the printed gate and who
                        # holds a value is 0.83 in the delivered file and the
                        # s3_11 branches overlap, so a declared-closed level
                        # can be one the CATI asked. A level whose observed
                        # skip rate is at least `recode_hi` is closed (its
                        # zeros become the sentinel); one at most `recode_lo`
                        # is open (they stay zeros); in between the label is
                        # ambiguous and those rows leave the fit.
                        ws = np.isin(wv, with_s) & keep
                        n_lev = len(cats[par]) + 1
                        cnt = np.bincount(pc_full[ws], minlength=n_lev)
                        sk = np.bincount(pc_full[ws & sent], minlength=n_lev)
                        rate = np.where(cnt >= int(CONFIG.get("recode_min", 20)),
                                        sk / np.maximum(cnt, 1),
                                        np.where(shut[:0].size == 0, np.nan, np.nan))
                        declared = ~opens_of[name]
                        rate = np.where(np.isnan(rate), declared.astype(float),
                                        rate)
                        hi, lo = float(CONFIG.get("recode_hi", 0.9)), \
                            float(CONFIG.get("recode_lo", 0.1))
                        r_row = rate[pc_full]
                        shut = r_row >= hi
                        ambiguous = (r_row > lo) & (r_row < hi)
                        keep = keep & ~(zero_rows & ambiguous & untouched)
                    y_all[zero_rows & shut & untouched] = width - 1
                    zero_q = np.isin(code[wave_col][query], without)
                    if not zero_q.any():
                        zero_q = None
            if wave_ok is not None and zero_q is None and not (
                    recode and gated and name in opens_of):
                restrict = wave_all
                if not restrict and gated:
                    sent = y_all == width - 1
                    here = bool(sent[wave_ok & keep].any())
                    there = bool(sent[(~wave_ok) & keep].any())
                    restrict = (here and not there) if wave_mode == "split" \
                        else (here != there)
                if restrict and int((wave_ok & keep).sum()) \
                        >= CONFIG["wave_floor"]:
                    keep = keep & wave_ok
            elif wave_ok is not None and wave_all:
                if int((wave_ok & keep).sum()) >= CONFIG["wave_floor"]:
                    keep = keep & wave_ok
            counts = np.bincount(y_all[keep], minlength=width).astype(
                np.float64) + CONFIG["smoothing"]
            crowd = counts / counts.sum()
            parts = ("crowd", np.tile(crowd, (len(query), 1)))

            def pooled(wclasses):
                """Answer vector over `wclasses` classes from one fit shared
                by the item's family (member index as an extra column);
                None when the item has no family or the fit failed."""
                fid = pool_family.get(name)
                if fid is None or select is not None:
                    return None
                members = pool_members[fid]
                key_ = (fid, tag, backbone, int(estimators), len(take),
                        _digest(take), wclasses, seed)
                if key_ not in pool_cache:
                    per = max(200, int(CONFIG.get("pool_rows_cap", 8000))
                              // len(members))
                    prng = np.random.default_rng(int(CONFIG["seed"]) + 9)
                    Xs, ys, ids = [], [], []
                    for mi, m in enumerate(members):
                        ym = code[m][take]
                        to_u, _nu = pool_maps[m]
                        rows_m = np.flatnonzero((ym >= 0) & (ym < len(to_u)))
                        if len(rows_m) > per:
                            rows_m = np.sort(prng.choice(rows_m, per,
                                                         replace=False))
                        Xs.append(Xtr_all[rows_m])
                        ys.append(to_u[ym[rows_m]])
                        ids.append(np.full(len(rows_m), float(mi), np.float32))
                    Xp = np.concatenate([np.concatenate(Xs),
                                         np.concatenate(ids)[:, None]], 1)
                    yp = np.concatenate(ys)
                    if np.unique(yp).size < 2:
                        pool_cache[key_] = None
                    else:
                        try:
                            m_ = fit_model(Xp, yp, estimators, backbone,
                                           list(cat_all) + [Xp.shape[1] - 1],
                                           int(CONFIG["seed"]) if seed is None
                                           else int(seed))
                            pool_cache[key_] = (m_, np.asarray(m_.classes_,
                                                               np.int64))
                        except Exception:
                            pool_cache[key_] = None
                if pool_cache[key_] is None:
                    return None
                m_, cls = pool_cache[key_]
                mi = float(members.index(name))
                Xq_p = np.concatenate([Xq_all, np.full((len(query), 1), mi,
                                                       np.float32)], 1)
                to_u, n_u = pool_maps[name]
                Pu = np.zeros((len(query), n_u))
                Pu[:, cls] = m_.predict_proba(Xq_p)
                P = np.zeros((len(query), wclasses))
                take_k = min(wclasses, len(to_u))
                P[:, :take_k] = Pu[:, to_u[:take_k]]
                return P / np.maximum(P.sum(1, keepdims=True), 1e-12)

            def shifted(mask, wclasses):
                """Answer vector over `wclasses` classes for an item that
                repeats a GIVEN question: fitted as the shift from the GIVEN
                answer in a shared label space, read back per row."""
                tw = twin_map.get(name)
                if tw is None or select is not None:
                    return None
                gi_, to_u_item, to_u_given, n_u = tw
                idx = np.flatnonzero(mask)
                xg = Xtr_all[idx, gi_].astype(np.int64)
                ug = to_u_given[np.minimum(xg, len(to_u_given) - 1)]
                ok = ug >= 0
                if int(ok.sum()) < 64:
                    return None
                yu = to_u_item[np.minimum(y_all[idx], len(to_u_item) - 1)]
                D = n_u - 1
                lab = (yu - ug + D)[ok]
                if np.unique(lab).size < 2:
                    return None
                key_ = ("delta", name, tag, backbone, int(estimators), len(take),
                        _digest(take), seed)
                if key_ not in pool_cache:
                    try:
                        m_ = fit_model(Xtr_all[idx[ok]], lab, estimators, backbone,
                                       cat_all, int(CONFIG["seed"]) if seed is None
                                       else int(seed))
                        pool_cache[key_] = (m_, np.asarray(m_.classes_, np.int64))
                    except Exception:
                        pool_cache[key_] = None
                if pool_cache[key_] is None:
                    return None
                m_, cls = pool_cache[key_]
                Pd = np.zeros((len(query), 2 * D + 1))
                Pd[:, cls] = m_.predict_proba(Xq_all)
                xq = Xq_all[:, gi_].astype(np.int64)
                uq = to_u_given[np.minimum(xq, len(to_u_given) - 1)]
                P = np.zeros((len(query), wclasses))
                rows_ok = np.flatnonzero(uq >= 0)
                take_k = min(wclasses, len(to_u_item))
                for k in range(take_k):
                    col = to_u_item[k] - uq[rows_ok] + D
                    P[rows_ok, k] = Pd[rows_ok, col]
                # rows whose GIVEN level has no counterpart: the crowd
                rest_rows = np.flatnonzero(uq < 0)
                if len(rest_rows):
                    c_ = np.bincount(y_all[idx], minlength=wclasses)[:wclasses].astype(float) + 0.5
                    P[rest_rows] = c_ / c_.sum()
                return P / np.maximum(P.sum(1, keepdims=True), 1e-12)

            def nested(mask, wclasses):
                """Answer vector over `wclasses` classes for the training rows
                in `mask`, with the non-response levels as their own hurdle.
                None when either side is too thin."""
                sp = spec_idx.get(name)
                if sp is None:
                    return None
                idx = np.flatnonzero(mask)
                ys = y_all[idx]
                is_sp = np.isin(ys, sp)
                if int(is_sp.sum()) < 32 or int((~is_sp).sum()) < 32:
                    return None
                pspec = fit_predict(Xtr[idx], is_sp.astype(np.int64), Xq, 2,
                                    est_sent, backbone, tag, cat_i, seed)[:, 1]
                pspec = np.clip(pspec, 1e-6, 1 - 1e-6)
                idm, ids = idx[~is_sp], idx[is_sp]
                if np.unique(y_all[idm]).size >= 2:
                    pmain = fit_predict(Xtr[idm], y_all[idm], Xq, wclasses,
                                        estimators, backbone, tag, cat_i, seed)
                else:
                    pmain = np.zeros((len(query), wclasses))
                    pmain[:, int(y_all[idm][0])] = 1.0
                if len(sp) >= 2 and np.unique(y_all[ids]).size >= 2:
                    psp = fit_predict(Xtr[ids], y_all[ids], Xq, wclasses,
                                      est_sent, backbone, tag, cat_i, seed)
                else:
                    c_ = np.bincount(y_all[ids], minlength=wclasses).astype(float)
                    psp = np.tile(c_ / c_.sum(), (len(query), 1))
                pmain = pmain / np.maximum(pmain.sum(1, keepdims=True), 1e-12)
                psp = psp / np.maximum(psp.sum(1, keepdims=True), 1e-12)
                return (1.0 - pspec)[:, None] * pmain + pspec[:, None] * psp
            try:
                if only is not None and name not in only:
                    pass
                elif time.time() > stop_at:
                    cut += 1
                elif width > max_width or not use_tabpfn:
                    parts = ("full", table_member(name, width, take, query))
                elif lat_on and name in lat_index:
                    # v38 lat: fit on (x, k_-j) over the query rows x every
                    # type, then integrate the joint over p(k | x); several
                    # decompositions average their joints. A gated item goes
                    # back as an exact hurdle so it mixes with base.
                    nq = len(query)
                    joints = []
                    decs = [(Kl, lat_loo(lat_index[name]), Pk, Xq_lat,
                             tag + "_lat%d" % Kl)
                            for Kl, lat_loo, Pk, Xq_lat in lat_dec]
                    # v38 lat_parent: a gated item whose parent is a PREDICT
                    # item takes the parent's own answer as its type and
                    # p(parent | x) as the integrating distribution -- the
                    # routing, learned per level and per x instead of prop's
                    # x-free skip table
                    lp_mode = (_pc("lat_parent_by_instrument", phase) or {}).get(key, CONFIG.get("lat_parent"))
                    par_ = gate[name]["parent"] if gated else None
                    if par_ is not None and not (par_ in code and records[par_]["class"] == "PREDICT"):
                        par_ = None
                    if lp_mode == "partner" and par_ is None:
                        # v39 partner: an item with no routing parent takes the
                        # item it shares the most information with
                        par_ = lat_partner.get(name)
                    if lp_mode and par_ is not None and par_ in code \
                            and records[par_]["class"] == "PREDICT":
                        Pp_ = parent_vector(par_, Xtr_all, Xq_all, cat_all)
                        kp_ = code[par_][take]
                        Kp_ = len(cats[par_])
                        if Pp_ is not None and (kp_ >= 0).all() \
                                and np.unique(kp_).size >= 2:
                            Pp_ = np.maximum(Pp_, 1e-9)
                            Pp_ = Pp_ / Pp_.sum(1, keepdims=True)
                            Xq_p_ = np.concatenate(
                                [np.repeat(Xq_all, Kp_, axis=0),
                                 np.tile(np.arange(Kp_, dtype=np.float32),
                                         len(Xq_all))[:, None]], axis=1)
                            pdec = (Kp_, kp_, Pp_, Xq_p_, tag + "_latp")
                            decs = [pdec] if lp_mode in ("replace", "only", "partner") \
                                else decs + [pdec]
                    for Kl, kj, Pk, Xq_lat, tg in decs:
                        kj = np.asarray(kj).astype(np.float32)
                        XtrL = np.concatenate([Xtr_all, kj[:, None]], axis=1)
                        if gated:
                            is_sent = (y_all == width - 1).astype(np.int64)
                            asked = keep & (y_all < width - 1)
                            if is_sent[keep].min() == is_sent[keep].max():
                                psk = np.full(len(Xq_lat), float(is_sent[keep].mean()))
                            else:
                                psk = fit_predict(XtrL[keep], is_sent[keep], Xq_lat, 2,
                                                  est_sent, backbone, tg,
                                                  cat_lat, seed)[:, 1]
                                fits += 1
                            if asked.sum() >= 32 and np.unique(y_all[asked]).size >= 2:
                                pak = fit_predict(XtrL[asked], y_all[asked], Xq_lat,
                                                  width - 1, estimators, backbone,
                                                  tg, cat_lat, seed)
                                fits += 1
                            else:
                                c2 = np.bincount(y_all[asked], minlength=width - 1) \
                                    + CONFIG["smoothing"]
                                pak = np.tile(c2 / c2.sum(), (len(Xq_lat), 1))
                            psk = np.clip(psk, 1e-6, 1 - 1e-6).reshape(nq, Kl)
                            pak = pak / np.maximum(pak.sum(1, keepdims=True), 1e-12)
                            pak = pak.reshape(nq, Kl, width - 1)
                            J = np.empty((nq, width))
                            J[:, :width - 1] = np.einsum("nk,nkc->nc", Pk * (1.0 - psk), pak)
                            J[:, width - 1] = np.einsum("nk,nk->n", Pk, psk)
                            joints.append(J)
                        elif keep.sum() >= 32 and np.unique(y_all[keep]).size >= 2:
                            Pl = fit_predict(XtrL[keep], y_all[keep], Xq_lat, width,
                                             estimators, backbone, tg, cat_lat, seed)
                            fits += 1
                            Pl = Pl / np.maximum(Pl.sum(1, keepdims=True), 1e-12)
                            joints.append(np.einsum("nk,nkc->nc", Pk,
                                                    Pl.reshape(nq, Kl, width)))
                    if joints:
                        J = np.mean(joints, axis=0)
                        J = J / np.maximum(J.sum(1, keepdims=True), 1e-12)
                        if gated:
                            ps = np.clip(J[:, width - 1], 1e-6, 1 - 1e-6)
                            pa = J[:, :width - 1] / np.maximum(
                                J[:, :width - 1].sum(1, keepdims=True), 1e-12)
                            if zero_q is not None:
                                z = zero_code.get(name, 0)
                                pa = pa.copy()
                                pa[zero_q] = pa[zero_q] * (1.0 - ps[zero_q])[:, None]
                                pa[zero_q, z] += ps[zero_q]
                                pa[zero_q] /= np.maximum(pa[zero_q].sum(1, keepdims=True), 1e-12)
                                ps = ps.copy()
                                ps[zero_q] = 1e-6
                            parts = ("hurdle", ps, pa)
                        else:
                            parts = ("full", J)
                elif gated:
                    is_sent = (y_all == width - 1).astype(np.int64)
                    asked = keep & (y_all < width - 1)
                    if CONFIG.get("exact_gates") and name in closed:
                        ps = closed[name][query].astype(np.float64)
                    elif is_sent[keep].min() == is_sent[keep].max():
                        ps = np.full(len(query), float(is_sent[keep].mean()))
                    else:
                        ps = fp(Xtr[keep], is_sent[keep], Xq, 2,
                                est_sent, backbone, tag, cat_i,
                                seed)[:, 1]
                        fits += 1
                    if prop and zero_q is None and select is None \
                            and name in parent_children.get(
                                gate[name]["parent"], ()):
                        # v25 prop: the parent's predicted vector through
                        # the routing table of the visible rows
                        par = gate[name]["parent"]
                        Pp = parent_vector(par, Xtr_all, Xq_all, cat_all)
                        if Pp is not None:
                            wp = len(cats[par])
                            yp = code[par][take]
                            both = keep & (yp >= 0)
                            cnt_p = np.bincount(yp[both], minlength=wp).astype(float)
                            sk_p = np.bincount(yp[both & (y_all == width - 1)],
                                               minlength=wp).astype(float)
                            r_p = (sk_p + 0.5) / (cnt_p + 1.0)
                            ps = Pp @ r_p
                    pa = nested(asked, width - 1) if nest else None
                    if pa is None and pool:
                        pa = pooled(width - 1)
                    if pa is None and delta:
                        pa = shifted(asked, width - 1)
                    if pa is not None:
                        fits += 2
                    elif asked.sum() >= 32 and np.unique(y_all[asked]).size >= 2:
                        pa = fp(Xtr[asked], y_all[asked], Xq,
                                width - 1, estimators, backbone, tag,
                                cat_i, seed)
                        fits += 1
                    else:
                        c2 = np.bincount(y_all[asked], minlength=width - 1) \
                            + CONFIG["smoothing"]
                        pa = np.tile(c2 / c2.sum(), (len(query), 1))
                    ps = np.clip(ps, 1e-6, 1 - 1e-6)
                    pa = pa / np.maximum(pa.sum(1, keepdims=True), 1e-12)
                    if zero_q is not None:
                        # that round's file holds a 0 where the routing
                        # skipped: the skip mass belongs to the 0 level
                        z = zero_code.get(name, 0)
                        pa = pa.copy()
                        pa[zero_q] = pa[zero_q] * (1.0 - ps[zero_q])[:, None]
                        pa[zero_q, z] += ps[zero_q]
                        pa[zero_q] /= np.maximum(pa[zero_q].sum(1, keepdims=True), 1e-12)
                        ps = ps.copy()
                        ps[zero_q] = 1e-6
                    parts = ("hurdle", ps, pa)
                elif keep.sum() >= 32 and np.unique(y_all[keep]).size >= 2:
                    P = nested(keep, width) if nest else None
                    if P is None and pool:
                        P = pooled(width)
                    if P is None and delta:
                        P = shifted(keep, width)
                    if P is not None:
                        fits += 2
                    else:
                        P = fp(Xtr[keep], y_all[keep], Xq, width,
                               estimators, backbone, tag, cat_i, seed)
                        fits += 1
                    parts = ("full", P / np.maximum(P.sum(1, keepdims=True),
                                                    1e-12))
            except Exception:
                parts = ("crowd", np.tile(crowd, (len(query), 1)))
            out[name] = (parts, width, crowd)
        return out, fits, cut

    def combine(passes, plan):
        """Per item, the weighted average of the passes the plan names, part
        by part. A pass that the clock cut (crowd) or whose part kind differs
        is left out and the weights renormalised, so an item whose secondary
        pass never ran keeps its primary vector exactly."""
        out = {}
        for name, parts in plan.items():
            have = [(p, w) for p, w in parts
                    if w > 0 and p in passes and name in passes[p]]
            if not have:
                primary = parts[0][0]
                if primary in passes and name in passes[primary]:
                    out[name] = passes[primary][name]
                continue
            kinds = [passes[p][name][0][0] for p, _ in have]
            kind = next((k for k in kinds if k != "crowd"), "crowd")
            use = [(p, w) for (p, w), k in zip(have, kinds) if k == kind]
            width, crowd = passes[use[0][0]][name][1], passes[use[0][0]][name][2]
            if kind == "crowd" or len(use) == 1:
                out[name] = passes[use[0][0]][name]
                continue
            tot = float(sum(w for _, w in use))
            if kind == "hurdle":
                ps = sum(w / tot * passes[p][name][0][1] for p, w in use)
                pa = sum(w / tot * passes[p][name][0][2] for p, w in use)
                out[name] = (("hurdle", np.clip(ps, 1e-6, 1 - 1e-6), pa),
                             width, crowd)
            else:
                P = sum(w / tot * passes[p][name][0][1] for p, w in use)
                out[name] = (("full", P), width, crowd)
        return out

    def split_rows(res, n_first):
        """A pass over concat(slice, out) -> (slice part, out part)."""
        a, b = {}, {}
        for name, (parts, width, crowd) in res.items():
            if parts[0] == "hurdle":
                pa_ = (("hurdle", parts[1][:n_first], parts[2][:n_first]),
                       width, crowd)
                pb_ = (("hurdle", parts[1][n_first:], parts[2][n_first:]),
                       width, crowd)
            else:
                pa_ = ((parts[0], parts[1][:n_first]), width, crowd)
                pb_ = ((parts[0], parts[1][n_first:]), width, crowd)
            a[name], b[name] = pa_, pb_
        return a, b

    def blend(a, b, wa, wb):
        """Two passes over the same rows -> their weighted mean per item;
        an item at the crowd in one pass takes the other pass alone."""
        out = {}
        for name in a:
            if b is None or name not in b:
                out[name] = a[name]
                continue
            (ka, *_), width, crowd = a[name]
            kb = b[name][0][0]
            if ka == "crowd" and kb != "crowd":
                out[name] = b[name]
            elif kb == "crowd" or ka != kb:
                out[name] = a[name]
            else:
                out[name] = combine({"a": {name: a[name]}, "b": {name: b[name]}},
                                    {name: [("a", wa), ("b", wb)]})[name]
        return out

    def mixed(parts, partner, w):
        """(1-w) parts + w partner; both are lists of (pass, weight)."""
        if w <= 0.0:
            return list(parts)
        out = [(p, wt * (1.0 - w)) for p, wt in parts]
        out += [(p, wt * w) for p, wt in partner]
        return out

    # ---- the gate: fit on the visible rows minus a slice, score the slice --
    E, Es = int(CONFIG["estimators"]), int(CONFIG["estimators_sentinel"])
    Eg = int(CONFIG["gate_estimators"])
    # v43: per instrument and phase. UNHCR's DEV block is one round, so the
    # container runs the recode path, where the gate's E4 base costs ~220 s
    # (rg2, which has no round-coded skips, 162 s) and the parent
    # integration no longer fits in the 900 s gate.
    _ge = (_pc("gate_estimators_by_instrument", phase) or {}).get(key)
    if _ge:
        Eg = int(_ge)
    # v40: the gate can be switched off per instrument (UNICEF: its gate was
    # measured at exactly 0 and its ~100 s go to the World Bank instead)
    gate_on = bool((_pc("gate_by_instrument", phase) or {}).get(key, CONFIG.get("gate"))) \
        and use_tabpfn and len(take_all) > 1500
    # The largest gate family: its sentinel gets its own calibration power.
    families = {}
    for name in targets:
        if gate[name]:
            families.setdefault(gate[name]["parent"], []).append(name)
    big = frozenset()
    if families:
        parent, members = max(families.items(), key=lambda kv: len(kv[1]))
        if len(members) >= int(CONFIG.get("big_family_min", 30)):
            big = frozenset(members)

    # A pass is one backbone over one feature set. A plan is, per item, a
    # weighted list of passes; the gate composes it in three moves -- the
    # feature view, then the item's own view where the slice separates it,
    # then a second backbone mixed in, instrument-wide and then per item.
    PASS_COST = dict(CONFIG["pass_cost"])

    def pass_kwargs_one(pname):
        if pname == "geo":
            return {"features": X_geo, "cat_idx": cat_geo}
        if pname == "all":
            return {"wave_all": True}
        if pname == "dup2":
            return {"dup_wave": True}
        if pname == "ordnum":
            return {"features": X_ord, "cat_idx": cat_ord}
        if pname == "roster":
            return {"features": X_roster}
        if pname == "onehot":
            return {"features": X_onehot, "cat_idx": list(cat_onehot)}
        if pname == "select":
            return {"select": select_map}
        if pname == "nest":
            return {"nest": True}
        if pname == "pool":
            return {"pool": True}
        if pname == "recode":
            return {"recode": True}
        if pname == "recode2":
            return {"recode": "emp"}
        if pname == "delta":
            return {"delta": True}
        if pname == "prop":
            return {"prop": True}
        if pname == "chain":
            return {"chain": True}
        if pname == "split":
            return {"split": True}
        if pname == "lat":
            return {"lat": True}
        if pname in PARTNER_BACKBONE:
            return {"backbone": PARTNER_BACKBONE[pname]}
        return {}

    def pass_kwargs(pname):
        # v26. `base_views` are views the container applies to EVERY pass,
        # including `base`, instead of offering them to the gate. The gate's
        # slice verdict is a per-run coin on a view whose fixture gain is
        # already known (recode: +0.0296 on sharp_w2); when the clock is tight
        # the verdict is not even reached and the shipped vectors silently
        # lose it. Anything listed here is decided here, not in the container.
        views = [v for v in CONFIG.get("base_views", ()) if v != pname
                 and pass_available(v)]
        if not views:
            return pass_kwargs_one(pname)
        conf = {}
        for v in views:
            conf.update(pass_kwargs_one(v))
        conf.update(pass_kwargs_one(pname))
        return conf

    def pass_available(pname):
        if pname == "geo":
            return X_geo is not None
        if pname in ("all", "dup2"):
            return wave_star is not None
        if pname == "ordnum":
            return X_ord is not None
        if pname == "roster":
            return X_roster is not None
        if pname == "onehot":
            return X_onehot is not None
        if pname == "select":
            return select_map is not None
        if pname == "nest":
            return bool(spec_idx)
        if pname == "pool":
            return bool(pool_family)
        if pname == "delta":
            return bool(twin_map)
        if pname == "prop":
            return bool(parent_children)
        if pname == "chain":
            return bool(chain_parents)
        if pname == "split":
            return split_col is not None
        if pname == "lat":
            return (bool(CONFIG.get("lat"))
                    and len(targets) >= int(CONFIG.get("lat_min_items", 8))
                    and phase in CONFIG.get("lat_phases", [1, 2])
                    and key in (_pc("lat_instruments", phase) or [key]))
        if pname == "ft" and phase not in CONFIG.get("ft_phases", [1, 2]):
            return False
        if pname in ("recode", "recode2"):
            return bool(CONFIG.get("recode")) and wave_col is not None and any(
                records[gate[n]["parent"]]["class"] == "PREDICT"
                for n in opens_of)
        if pname == "exa" and key not in CONFIG.get("exa_instruments", [key]):
            return False
        if pname == "tfm" and key not in CONFIG.get("tfm_instruments", [key]):
            return False
        if pname == "kumo" and key not in CONFIG.get("kumo_instruments", [key]):
            return False
        if pname == "cau" and key not in CONFIG.get("cau_instruments", [key]):
            return False
        if pname in PARTNER_BACKBONE:
            return partner_spec(PARTNER_BACKBONE[pname])[0] is not None
        return pname == "base"

    CPU_PASSES = [c for c in CONFIG.get("cpu_passes", [])
                  if key in CONFIG.get("cpu_instruments", [])]
    PARTNERS = set(PARTNER_BACKBONE) | set(CPU_PASSES)
    worker_py = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "cpu_worker.py")
    if CPU_PASSES and not os.path.exists(worker_py):
        CPU_PASSES = []

    def cpu_request(take, query, stop_at):
        """Everything cpu_worker.py needs, with the per-item row masks the
        GPU pass would use on the same rows (round restriction included)."""
        wave_ok = None
        if wave_star is not None:
            wave_ok = code[wave_col][take] == wave_star
            if int(wave_ok.sum()) < int(CONFIG["wave_floor"]):
                wave_ok = None
        rows = []
        for name in targets:
            width = len(cats[name])
            y_all = code[name][take]
            keep = y_all >= 0
            gated = bool(gate[name])
            if wave_ok is not None and gated:
                sent = y_all == width - 1
                here = bool(sent[wave_ok & keep].any())
                there = bool(sent[(~wave_ok) & keep].any())
                restrict = (here and not there) if wave_mode == "split" \
                    else (here != there)
                if restrict and int((wave_ok & keep).sum()) \
                        >= CONFIG["wave_floor"]:
                    keep = keep & wave_ok
            rows.append({"name": name, "width": width, "gated": gated,
                         "y": y_all.astype(np.int32), "keep": keep})
        return {"X": X, "widths": np.asarray([len(cats[g]) for g in given]),
                "take": take, "query": query, "items": rows,
                "kinds": list(CPU_PASSES), "deadline": float(stop_at),
                "hp": {"hgb": CONFIG.get("hgb", {}),
                       "donor": CONFIG.get("donor", {}),
                       "lr2_C": CONFIG.get("lr2_C", 0.5),
                       "lr2_pair_max_levels": CONFIG.get("lr2_pair_max_levels", 12),
                       "lr2_min_support": CONFIG.get("lr2_min_support", 5),
                       "variants": CONFIG.get("cpu_variants", {}),
                       "smoothing": CONFIG["smoothing"],
                       "seed": CONFIG["seed"]}}

    def cpu_start(take, query, stop_at):
        """Launch the worker; returns (process, result path) or None."""
        if not CPU_PASSES:
            return None
        import pickle
        import subprocess
        import sys as _sys
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="sb_cpu_")
        req_path = os.path.join(tmpdir, "request.pkl")
        out_path = os.path.join(tmpdir, "result.pkl")
        with open(req_path, "wb") as fh:
            pickle.dump(cpu_request(take, query, stop_at), fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        env = dict(os.environ)
        thr = str(int(CONFIG.get("cpu_threads", 3)))
        env.update({"OMP_NUM_THREADS": thr, "OPENBLAS_NUM_THREADS": thr,
                    "MKL_NUM_THREADS": thr, "PYTHONUNBUFFERED": "1"})
        try:
            proc = subprocess.Popen([_sys.executable, worker_py, req_path,
                                     out_path], env=env,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE)
        except Exception as exc:
            say("cpu worker failed to start: %r" % (exc,))
            return None
        return (proc, out_path, time.time())

    def cpu_collect(handle, n_first, wait):
        """Wait up to `wait` seconds for the worker; returns
        {kind: (slice part, out part)} in run_pass format."""
        if handle is None:
            return {}, {}
        import pickle
        proc, out_path, t_start = handle
        end = time.time() + max(0.0, float(wait))
        while not os.path.exists(out_path) and time.time() < end:
            if proc.poll() is not None and not os.path.exists(out_path):
                break
            time.sleep(0.5)
        if not os.path.exists(out_path):
            try:
                proc.kill()
            except Exception:
                pass
            err = b""
            try:
                err = proc.stderr.read()[-400:] if proc.stderr else b""
            except Exception:
                pass
            say("cpu worker not finished after %.0fs; dropped (%s)"
                % (time.time() - t_start, err.decode("utf-8", "ignore")
                   .strip().replace("\n", " | ")))
            return {}, {}
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        with open(out_path, "rb") as fh:
            got = pickle.load(fh)
        out, info = {}, {"timing": got.get("timing"), "done": got.get("done"),
                         "design_cols": got.get("design_cols"),
                         "seconds": round(time.time() - t_start, 1),
                         "errors": len(got["result"].get("_errors", []))}
        for kind in CPU_PASSES:
            res = {}
            for name, parts in got["result"].get(kind, {}).items():
                if parts[0] == "hurdle":
                    _, ps, pa, width, crowd = parts
                    res[name] = (("hurdle", ps, pa), width, crowd)
                else:
                    _, P, width, crowd = parts
                    res[name] = (("full", P), width, crowd)
            if res:
                out[kind] = split_rows(res, n_first) if n_first is not None \
                    else (res, None)
        return out, info

    plan = {name: [("base", 1.0)] for name in targets}
    gate_out = {} if CONFIG.get("gate_out", True) else None
    cal, floor = (1.0, 1.0, 1.0, 1.0), float(CONFIG["floor"])
    report = {"instrument": key, "gate": False, "wave": wave_star,
              "unseen": unseen_note,
              "select_items": 0 if select_map is None else len(select_map),
              "nest_items": len(spec_idx),
              "pool_items": len(pool_family), "pool_families": len(pool_members),
              "delta_items": len(twin_map),
              "prop_items": sum(len(v) for v in parent_children.values()),
              "chain_parents": list(chain_parents),
              "split_col": given[split_col] if split_col is not None else None,
              "select_mean_cols": 0 if select_map is None else round(float(
                  np.mean([len(v) for v in select_map.values()])), 2),
              "geo_cols": 0 if geo_cols is None else int(geo_cols.shape[1]),
              "ord_cols": ord_cols, "big_family": len(big)}
    base_cost, base_cut = None, 0
    if gate_on:
        want = CONFIG["gate_rows"]
        want = int(want.get(key, 600)) if isinstance(want, dict) else int(want)
        pool = take_all
        if wave_star is not None:
            pool = take_all[code[wave_col][take_all] == wave_star]
        want = min(want, len(pool) // 3)
        grng = np.random.default_rng(CONFIG["seed"] + 1)
        # v29: in the final phase the DEV respondents are visible, answers
        # included, and they are the rows the development phase scored --
        # assigned once when the dataset was built, by position, so they are
        # the last `n_dev` visible rows of the delivered frame. Whatever the
        # file is sorted by, they are the visible population nearest to the
        # TEST block, so the judge (calibration, candidates, per-item
        # weights) sits on them instead of a draw from every visible row.
        # A shuffled file makes them a plain random draw: nothing is lost.
        dev_val = None
        if CONFIG.get("dev_slice") and phase == 2 and split.get("n_dev"):
            n_dev_ = int(split["n_dev"])
            block = fit_rows[-n_dev_:] if len(fit_rows) > n_dev_ else fit_rows
            block = np.intersect1d(block, pool)
            cap_ = int(CONFIG.get("dev_slice_max", 1500))
            if len(block) > cap_:
                block = np.sort(grng.choice(block, cap_, replace=False))
            if len(block) >= max(100, want // 2):
                dev_val = block
        # the most shifted GIVEN column between the pool and the held-out
        # rows stratifies the draw
        strata, tv_max = None, 0.0
        for name in given:
            if name == wave_col and wave_star is not None:
                continue
            W = len(cats[name]) + 1
            a = code[name][pool]
            b = code[name][out_rows]
            pa = np.bincount(np.where(a < 0, W - 1, a), minlength=W) / max(len(a), 1)
            pb = np.bincount(np.where(b < 0, W - 1, b), minlength=W) / max(len(b), 1)
            tv = 0.5 * float(np.abs(pa - pb).sum())
            if tv > tv_max:
                strata, tv_max = name, tv
        report_slice = {"slice_col": strata, "slice_tv": round(tv_max, 3),
                        "slice_strat": False}
        if dev_val is None and strata is not None \
                and tv_max >= float(CONFIG.get("slice_shift_tv", 1.1)) \
                and len(out_rows):
            W = len(cats[strata]) + 1
            a = code[strata][pool]
            a = np.where(a < 0, W - 1, a)
            b = code[strata][out_rows]
            b = np.where(b < 0, W - 1, b)
            have = np.bincount(a, minlength=W)
            wantp = np.bincount(b, minlength=W).astype(np.float64)
            wantp[have == 0] = 0.0
            if wantp.sum() > 0:
                wantp /= wantp.sum()
                cap = (have * 0.5).astype(np.int64)
                alloc = np.zeros(W, np.int64)
                left = int(want)
                open_ = wantp > 0
                for _ in range(W):
                    if left <= 0 or not open_.any():
                        break
                    share_ = wantp * open_
                    share_ /= share_.sum()
                    give = np.floor(share_ * left).astype(np.int64)
                    give = np.minimum(give, cap - alloc)
                    alloc += give
                    left -= int(give.sum())
                    open_ &= alloc < cap
                    if give.sum() == 0:
                        break
                picks = []
                for lvl in np.flatnonzero(alloc > 0):
                    rows_ = pool[a == lvl]
                    picks.append(grng.choice(rows_, int(alloc[lvl]), replace=False))
                if picks and sum(len(p_) for p_ in picks) >= max(100, want // 2):
                    val = np.sort(np.concatenate(picks))
                    report_slice["slice_strat"] = True
                    report_slice["slice_levels"] = int((alloc > 0).sum())
        if dev_val is not None:
            val = dev_val
            want = len(val)
            report_slice.update({"slice_strat": False, "slice_dev": True})
        elif not report_slice["slice_strat"]:
            val = np.sort(grng.choice(pool, want, replace=False))
        rest = np.setdiff1d(take_all, val)
        truth = {name: code[name][val] for name in targets}
        gs = _pc("gate_share", phase)  # v41: phase-aware
        gs = float(gs.get(key, 0.55)) if isinstance(gs, dict) else float(gs)
        gate_end = time.time() + (deadline - time.time()) * gs
        # ---- v24: fine-tune the primary on the rows the slice does not hold ----
        if use_tabpfn and key in (_pc("ft_instruments", phase) or []) \
                and "ft" in CONFIG["candidates"] and path \
                and phase in CONFIG.get("ft_phases", [1, 2]):
            fs = CONFIG.get("ft_share", 0.35)
            fs = float(fs.get(str(phase), 0.35)) if isinstance(fs, dict) else float(fs)
            ft_stop = time.time() + (gate_end - time.time()) * fs
            try:
                spec_ft, ft_info = _finetune(
                    path, X[rest], categorical,
                    [code[nm][rest] for nm in targets], ft_stop,
                    int(CONFIG["seed"]) + 11, say,
                    gated=[len(cats[nm]) if gate[nm] else 0 for nm in targets])
            except Exception as exc:
                spec_ft, ft_info = None, {"error": type(exc).__name__}
            if spec_ft is not None:
                ft_holder["spec"] = spec_ft
            report["ft"] = ft_info
            say("fine-tune: %s" % (ft_info,))
        # ---- 10/9: fine-tune the Kumo partner on the same rows ----
        kft_on = (_pc("kumo_ft_instruments", phase) or [])
        if use_tabpfn and key in kft_on and "kumo" in CONFIG["candidates"] \
                and key in CONFIG.get("kumo_instruments", [key]) \
                and partner_spec("kumo")[0] is not None:
            ks = _pc("kumo_ft_share", phase) or {}
            ks = float(ks.get(key, 0.2)) if isinstance(ks, dict) else float(ks)
            k_stop = time.time() + (gate_end - time.time()) * ks
            # 10/14: keep room in the gate for the Kumo pass itself (U3 sbx: ft ate the gate, kumo pass skipped)
            k_res = (lambda r_: float(r_.get(key, 0.0)) if isinstance(r_, dict) else float(r_ or 0.0))(_pc("kumo_ft_reserve", phase))
            k_stop = min(k_stop, gate_end - k_res) if k_res > 0 else k_stop
            try:
                kft_info = _kumo_finetune(partner_spec("kumo")[0], X[rest],
                                          [code[nm][rest] for nm in targets], k_stop,
                                          int(CONFIG["seed"]) + 13, say,
                                          lr=(lambda l_: l_.get(key, 2e-05) if isinstance(l_, dict) else l_)(_pc("kumo_ft_lr", phase)), cap=(lambda c_: c_.get(key, 600) if isinstance(c_, dict) else c_)(_pc("kumo_ft_steps", phase)))
            except Exception as exc:
                kft_info = {"error": repr(exc)[:200]}
                try:
                    p_k = partner_spec("kumo")[0]
                    next(iter(_KUMO[p_k].models.values())).load_state_dict(
                        {k: v.to("cuda") for k, v in _KUMO["base_sd:" + p_k].items()})
                    _KUMO[p_k].eval()
                except Exception:
                    pass
            report["kumo_ft"] = kft_info
            say("kumo fine-tune: %s" % (kft_info,))
        t0 = time.time()
        sf = float(CONFIG["scorer_floor"])
        # The gate passes predict the held-out rows too (a few percent more
        # query rows): those vectors, fitted on the visible rows minus the
        # slice with their own seed, join the final pass as extra members,
        # and stand in for it wherever the clock cuts the final.
        gate_query, gate_seed = val, None
        if CONFIG.get("gate_out", True):
            gate_query = np.concatenate([val, out_rows])
            gate_seed = int(CONFIG["seed"]) + 7

        # ---- the CPU family, in its own interpreter, for the whole gate ----
        cpu_handle = cpu_start(rest, gate_query, gate_end)
        # ---- the GPU passes, in priority order, while the gate has time ----
        order_ = ["base"] + [("v2" if c == "v2mix" else c)
                             for c in CONFIG["candidates"]
                             if c not in CPU_PASSES]
        passes, spent = {}, {}
        skip_ = set((_pc("skip_passes", phase) or {}).get(key, []))
        for pname in order_:
            if pname in passes or not pass_available(pname):
                continue
            if pname in skip_:
                say("pass %s skipped by the phase plan" % pname)
                continue
            costf = float(PASS_COST.get(pname, 1.0))
            if pname == "lat":
                costf *= max(1, len(_pc("lat_ks", phase) or [1]))
                # v42: the static estimate was measured at K = 8; the query rows
                # (and so the pass) scale with the class count
                costf *= max(int(k_) for k_ in (_pc("lat_ks", phase)
                             or [int(_pc("lat_k", phase, 8))])) / 8.0
                if (_pc("lat_parent_by_instrument", phase) or {}).get(key, CONFIG.get("lat_parent")) in ("only", "partner"):
                    # the parent-level expansion (3-6 levels) on the gated
                    # items with a PREDICT parent only; per item about 1.6 base
                    # fits (rg2: 144 items at E2 in 86 s against a 161 s base)
                    n_pp = sum(1 for n_ in targets if gate[n_]
                               and gate[n_]["parent"] in records
                               and records[gate[n_]["parent"]]["class"] == "PREDICT")
                    if (_pc("lat_parent_by_instrument", phase) or {}).get(key, CONFIG.get("lat_parent")) == "partner":
                        n_pp = len(targets)
                    costf = float(CONFIG.get("lat_parent_cost", 1.6)) * max(
                        n_pp, 1) / float(max(len(targets), 1))
                if _pc("lat_gate_estimators", phase):
                    costf *= min(Eg, int(_pc("lat_gate_estimators", phase))) / float(Eg)
            only_ = None
            cheap = pname not in CONFIG.get("full_candidates", ()) \
                and pname != "base" and "base" in passes
            if cheap:
                # v22: every candidate pass is judged on the items the base
                # pass scores worst on the slice, up to `gate_items_max` per
                # instrument (TabFM: `tfm_items_max`, narrow items only), and
                # the final pass names those items only. The whole candidate
                # list then gets its slice verdict inside the 900 s clock
                # instead of the first three passes.
                if pname == "tfm":
                    kmax = int((CONFIG.get("tfm_items_max") or {}).get(key, 0) or 0)
                    wmax = int(CONFIG.get("tfm_max_width", 10))
                else:
                    kmax = int((CONFIG.get("gate_items_max") or {}).get(key, 0) or 0)
                    wmax = 10 ** 6
                ranked = []
                for n_, (parts_, width_, crowd_) in passes["base"].items():
                    if width_ > wmax:
                        continue
                    vec_ = _compose(parts_, width_, (1.0, 1.0, 1.0, 1.0),
                                    crowd_, float(CONFIG["floor"]), n_ in big)
                    y_ = code[n_][val]
                    ok_ = y_ >= 0
                    if not ok_.any():
                        continue
                    ranked.append((-float(np.log(vec_[ok_, y_[ok_]]).mean()), n_))
                ranked.sort(reverse=True)
                if kmax and kmax < len(ranked):
                    only_ = set(n_ for _, n_ in ranked[:kmax])
                    costf *= float(kmax) / max(len(targets), 1)
                else:
                    only_ = set(n_ for _, n_ in ranked)
                    costf *= float(len(ranked)) / max(len(targets), 1)
            if pname != "base" and time.time() + base_cost * costf > gate_end:
                say("pass %s skipped: %.0fs needed, %.0fs left in gate"
                    % (pname, base_cost * costf, gate_end - time.time()))
                continue
            t1 = time.time()
            Eg_p = Eg
            if pname == "lat" and _pc("lat_gate_estimators", phase):
                Eg_p = min(Eg, int(_pc("lat_gate_estimators", phase)))
            res, fits, cut = run_pass(rest, gate_query, Eg_p, Eg_p, tag="gate",
                                      stop_at=gate_end, seed=gate_seed,
                                      only=only_, **pass_kwargs(pname))
            spent[pname] = round(time.time() - t1, 1)
            if gate_out is not None:
                res, gate_out[pname] = split_rows(res, len(val))
            if pname == "base":
                base_cost, base_cut = time.time() - t1, int(cut)
                if base_cut:
                    say("base pass cut by the clock (%d items): no decisions"
                        % base_cut)
                    break
            elif cut:
                say("pass %s cut by the clock (%d items); ignored"
                    % (pname, cut))
                continue
            passes[pname] = res

        # the CPU family: whatever the worker finished by the gate's end
        cpu_got, cpu_info = cpu_collect(
            cpu_handle, len(val) if gate_out is not None else None,
            max(0.0, gate_end - time.time()) + float(CONFIG.get("cpu_wait", 20.0)))
        if cpu_info:
            report["cpu"] = cpu_info
            say("cpu worker: %s" % (cpu_info,))
        for pname, (res_a, res_b) in cpu_got.items():
            if base_cut:
                break
            passes[pname] = res_a
            spent[pname] = 0.0
            if gate_out is not None:
                gate_out[pname] = res_b
        # the measured cost of every pass that ran replaces the table's
        # guess, so the final pass is sized from this box's own clock
        for pname, secs in spent.items():
            if pname in passes and base_cost and pname not in CPU_PASSES:
                PASS_COST[pname] = max(0.2, float(secs) / float(base_cost))
                if pname == "lat" and _pc("lat_gate_estimators", phase):
                    # measured at its own gate E; the table is per gate E
                    PASS_COST[pname] *= float(Eg) / min(Eg, int(_pc("lat_gate_estimators", phase)))
        for pname in CPU_PASSES:
            PASS_COST[pname] = 0.0
        uniform = float(np.mean([np.log(len(cats[t])) for t in targets]))
        report.update(report_slice)
        report.update({"gate": True, "slice": int(len(val)),
                       "base_cut": base_cut, "base_cost": round(base_cost, 1),
                       "passes": spent, "views": {}, "mixes": {}})
        if not base_cut:
            base = passes["base"]
            has_h = any(v[0][0] == "hurdle" for v in base.values())
            has_f = any(v[0][0] == "full" for v in base.values())
            raw_loss = _slice_loss(base, truth, (1.0, 1.0, 1.0, 1.0),
                                   float(CONFIG["floor"]), sf, big)
            # v30: per-instrument margin. On the 3.5 backbone the World Bank
            # slice base is +0.0033 over v3m and the fine-tune's +0.0014 on
            # top fell under 0.0015 (rgw_real 0.4944 vs 0.4951 at 0.0008).
            margin = float((CONFIG.get("accept_margin_by_instrument") or {})
                           .get(key, CONFIG["accept_margin"]))

            def judge(res):
                if CONFIG.get("calibrate"):
                    c_cal, c_floor, loss = _fit_calibration(res, truth, has_h,
                                                            has_f, big)
                else:
                    c_cal, c_floor = (1.0, 1.0, 1.0, 1.0), float(CONFIG["floor"])
                    loss = _slice_loss(res, truth, c_cal, c_floor, sf, big)
                return 1 - loss / uniform, c_cal, c_floor

            def judge_plan(pl):
                return judge(combine(passes, pl))

            def uniform_plan(parts):
                """The plan `parts` on every item; an item a named pass did
                not reach (the gate judges a candidate on its worst-K items
                only) keeps the base pass, so a view is judged on the items
                it covers and not on holes."""
                plan_ = {}
                for n in targets:
                    ok = all(w <= 0 or (p in passes and n in passes[p])
                             for p, w in parts)
                    plan_[n] = list(parts) if ok else [("base", 1.0)]
                return plan_

            def log_rows(res, name, c_cal, c_floor):
                parts, width, crowd = res[name]
                P = _compose(parts, width, c_cal, crowd, c_floor, name in big)
                P = sf + (1.0 - width * sf) * P
                y = truth[name]
                return np.log(P[np.arange(len(y)), y])

            def pick_items(pl, options, c_cal, c_floor, z, seed):
                """Per item, the option that beats the item's current plan
                by more than z respondent-bootstrap standard errors on the
                slice. `options(name)` -> {label: parts}."""
                rng_ = np.random.default_rng(seed)
                new, moved = {}, 0
                for name in pl:
                    cur = combine(passes, {name: pl[name]})
                    if name not in cur:
                        new[name] = pl[name]
                        continue
                    base_row = log_rows(cur, name, c_cal, c_floor)
                    best, best_gain, best_diff = None, 0.0, None
                    for label, parts in options(name).items():
                        if list(parts) == list(pl[name]):
                            continue
                        alt = combine(passes, {name: parts})
                        if name not in alt:
                            continue
                        diff = log_rows(alt, name, c_cal, c_floor) - base_row
                        gain = float(diff.mean())
                        if gain > best_gain:
                            best, best_gain, best_diff = parts, gain, diff
                    if best is not None:
                        picks = rng_.integers(0, len(best_diff),
                                              size=(200, len(best_diff)))
                        se = float(best_diff[picks].mean(1).std(ddof=1))
                        if best_gain > z * se:
                            new[name] = list(best)
                            moved += 1
                            continue
                    new[name] = pl[name]
                return new, moved

            # ---- 1. the feature view, instrument-wide ----
            view_plans = {"base": [("base", 1.0)]}
            for v in ("recode2", "recode", "prop", "chain", "split", "delta",
                      "geo", "all", "dup2", "ordnum", "roster", "onehot",
                      "select", "nest", "pool", "lat"):
                if v in passes:
                    view_plans[v] = [(v, 1.0)]
            for v in ("recode2", "recode", "prop", "chain", "split", "delta",
                      "geo", "dup2", "ordnum", "roster", "onehot", "select",
                      "nest", "pool", "lat"):
                if v in passes:
                    view_plans["base+" + v] = [("base", 0.5), (v, 0.5)]
            verdict = {label: judge_plan(uniform_plan(parts))
                       for label, parts in view_plans.items()}
            base_skill = verdict["base"][0]
            b_cal, b_floor = verdict["base"][1], verdict["base"][2]
            best_label, best_skill = "base", base_skill
            # v25: a candidate pass runs on its worst-K items only, so its
            # verdict is taken on the items every named pass reached, with
            # the base calibration held fixed -- judged on the whole
            # instrument, a +0.003 on a third of the items reads as +0.001
            # and never clears the margin. `view_margin` is that margin on
            # the covered items' own scale; the item-level z test is
            # unchanged.
            v_margin = float(CONFIG.get("view_margin", margin))
            res_base = combine(passes, uniform_plan(view_plans["base"]))
            report["views_cover"] = {}
            best_gain = 0.0
            for label, parts in view_plans.items():
                if label == "base":
                    continue
                cover = set(targets)
                for p_, w_ in parts:
                    if w_ > 0:
                        cover &= {n for n in passes.get(p_, {})
                                  if passes[p_][n][0][0] != "crowd"}
                if not cover:
                    continue
                res_v = combine(passes, uniform_plan(parts))
                sub_v = {n: res_v[n] for n in cover if n in res_v}
                sub_b = {n: res_base[n] for n in cover if n in res_base}
                if not sub_v or len(sub_v) != len(sub_b):
                    continue
                gain = (_slice_loss(sub_b, truth, b_cal, b_floor, sf, big)
                        - _slice_loss(sub_v, truth, b_cal, b_floor, sf, big)) / uniform
                report["views_cover"][label] = (len(cover), round(gain, 4))
                if gain > v_margin and gain > best_gain:
                    best_label, best_gain = label, gain
            if best_label != "base":
                best_skill = verdict[best_label][0]
                if best_skill < base_skill:      # never below the base plan
                    best_label, best_skill = "base", base_skill
            for label in view_plans:
                skill = verdict[label][0]
                report["views"][label] = round(skill, 4)
                cov = report["views_cover"].get(label)
                say("view %-14s skill %.4f (%+.4f vs base)  cal=%s floor=%g%s"
                    % (label, skill, skill - base_skill, verdict[label][1],
                       verdict[label][2],
                       "  cover %d items %+.4f" % cov if cov else ""))
            cal, floor = verdict[best_label][1], verdict[best_label][2]
            plan = uniform_plan(view_plans[best_label])
            say("raw base %.4f -> calibrated %.4f; view %s at %.4f"
                % (1 - raw_loss / uniform, base_skill, best_label, best_skill))
            report.update({"skill_raw": round(1 - raw_loss / uniform, 4),
                           "skill_base": round(base_skill, 4),
                           "view": best_label})

            # ---- 2. the item's own view, where the slice separates it ----
            if CONFIG.get("item_view") and len(view_plans) > 1:
                plan, moved = pick_items(plan, lambda name: view_plans, cal,
                                         floor, float(CONFIG["view_z"]),
                                         CONFIG["seed"] + 3)
                if moved:
                    best_skill, cal, floor = judge_plan(plan)
                say("item views: %d of %d items leave %s, slice skill %.4f "
                    "(in-sample)" % (moved, len(plan), best_label, best_skill))
                report["item_view_moved"] = moved

            # ---- 3. a second backbone mixed in ----
            partner_plans = {}
            avail = [p for p in CONFIG["mix_partners"] if p in passes]
            avail += [p for p in CPU_PASSES if p in passes and p not in avail]
            single_skill = {}
            for p in avail:
                partner_plans[p] = [(p, 1.0)]
                trial = {n: mixed(plan[n], [(p, 1.0)], 0.5) for n in targets}
                single_skill[p] = judge_plan(trial)[0]
            top_ = sorted(avail, key=lambda p: -single_skill[p])[
                :int(CONFIG.get("partner_top", 3))]
            for i_ in range(len(top_)):
                for j_ in range(i_ + 1, len(top_)):
                    pair_ = [top_[i_], top_[j_]]
                    partner_plans["+".join(pair_)] = [(p, 0.5) for p in pair_]
            if len(top_) >= 3:
                partner_plans["+".join(top_)] = [(p, 1.0 / len(top_))
                                                 for p in top_]
            w0, partner = 0.0, None
            view_skill = best_skill
            top = None
            for label, pp in partner_plans.items():
                # v40 partner_grid: with more fine-tune passes on the clock the
                # partner may deserve the whole weight; default unchanged
                for w in tuple((_pc("partner_grid_by_instrument", phase) or {}).get(key)
                               or _pc("partner_grid", phase) or (0.25, 0.5, 0.75)):
                    trial = {n: mixed(plan[n], pp, w) for n in targets}
                    skill, c_cal, c_floor = judge_plan(trial)
                    report["mixes"]["%s@%g" % (label, w)] = round(skill, 4)
                    say("mix %-8s@%-5g skill %.4f (%+.4f vs view)  cal=%s "
                        "floor=%g" % (label, w, skill, skill - view_skill,
                                      c_cal, c_floor))
                    npass = len([p for p, _ in pp if p not in CPU_PASSES])
                    # a partner that costs one more final pass must beat
                    # the best cheaper one by `pair_margin`, not merely tie
                    if top is None or skill > top[0] + (
                            float(CONFIG.get("pair_margin", 0.0005))
                            if npass > top[5] else 0.0) \
                            or (npass < top[5] and skill > top[0] - float(
                                CONFIG.get("pair_margin", 0.0005))):
                        top = (skill, label, w, c_cal, c_floor, npass)
            # the best mix, if it clears the margin over the view itself
            if top is not None and top[0] > view_skill + margin:
                best_skill, partner, w0, cal, floor = top[:5]
            if partner_plans:
                if partner is None:
                    partner = max(partner_plans, key=lambda l: report["mixes"]
                                  .get("%s@0.5" % l, -1.0))
                pp = partner_plans[partner]
                plan = {n: mixed(plan[n], pp, w0) for n in targets}
                if CONFIG.get("item_mix"):
                    grid = list(CONFIG["mix_grid"])
                    pre = {n: list(v) for n, v in plan.items()}
                    view_of = {n: [(p, wt) for p, wt in pre[n]
                                   if p not in PARTNERS] for n in targets}

                    def weight_options(name, _v=view_of, _pp=pp, _g=grid):
                        parts = _v[name]
                        tot = sum(wt for _, wt in parts) or 1.0
                        parts = [(p, wt / tot) for p, wt in parts]
                        return {"w%g" % w: mixed(parts, _pp, w) for w in _g}

                    plan, moved = pick_items(plan, weight_options, cal, floor,
                                             float(CONFIG["item_z"]),
                                             CONFIG["seed"] + 2)
                    m_loss = _slice_loss(combine(passes, plan), truth, cal,
                                         floor, sf, big)
                    say("item weights: %d of %d items leave w=%g (%s), slice "
                        "skill %.4f (in-sample)"
                        % (moved, len(plan), w0, partner, 1 - m_loss / uniform))
                    report["item_moved"] = moved
            # ---- 4. v38: the latent-type pass mixed in on a veto ----
            # The real-survey ruler already priced it (+0.0005..+0.0009 at
            # 0.5 on all three shapes); the slice only refuses a clear loss.
            # The calibration is re-chosen on the mixed plan.
            # v41: the mix rule may differ per instrument (UNHCR takes the parent
            # integration whole at 900 s; the World Bank latent type wants a grid)
            lm = (_pc("lat_mix_by_instrument", phase) or {}).get(key) \
                or _pc("lat_mix", phase) or {}
            if lm and "lat" in passes:
                have_l = {n for n in passes["lat"]
                          if passes["lat"][n][0][0] != "crowd"}
                # the weight: `w`, or the slice's best of `grid`
                t_best = None
                for w_ in (lm.get("grid") or [lm.get("w", 0.5)]):
                    trial_ = {n: (mixed(plan[n], [("lat", 1.0)], float(w_))
                                  if n in have_l else plan[n]) for n in targets}
                    r_ = judge_plan(trial_)
                    report.setdefault("lat_grid", {})["%g" % float(w_)] = round(
                        r_[0] - best_skill, 5)
                    if t_best is None or r_[0] > t_best[0]:
                        t_best = (r_[0], r_[1], r_[2], trial_, float(w_))
                t_skill, t_cal, t_floor, trial, w_l = t_best
                # accept on a slice gain of at least `min_gain` (v38b: +0.0002;
                # the v38 veto of -0.0005 let rgw_real's -0.0001 through and the
                # final pass then lost E8 -> E5 for nothing, -0.0005)
                ok_l = t_skill - best_skill >= float(
                    lm.get("min_gain", -float(lm.get("veto", 0.0005))))
                say("lat mix @%g on %d items: slice skill %.4f (%+.4f), %s"
                    % (w_l, len(have_l), t_skill, t_skill - best_skill,
                       "taken" if ok_l else "vetoed"))
                report["lat_mix"] = (len(have_l), round(t_skill - best_skill, 4),
                                     bool(ok_l))
                if ok_l:
                    plan, best_skill, cal, floor = trial, t_skill, t_cal, t_floor
            counts = {}
            for n, parts in plan.items():
                for p, w in parts:
                    if w > 0:
                        counts[p] = counts.get(p, 0) + 1
            report.update({"winner": "%s|%s@%s" % (best_label, partner, w0),
                           "cal": cal, "floor": floor,
                           "skill_gate": round(best_skill, 4),
                           "plan_counts": counts})
        report["gate_seconds"] = round(time.time() - t0, 1)
        if os.environ.get("SB_GATE_DUMP") and not base_cut:
            import pickle
            with open(os.environ["SB_GATE_DUMP"] + "." + key + ".pkl", "wb") as fh:
                pickle.dump({"base": passes["base"], "truth": truth,
                             "passes": passes, "plan": plan, "report": report,
                             "cats": cats, "gate": gate, "big": big,
                             "cal": cal, "floor": floor}, fh)

        # ---- v28: the partners every item takes, judged nowhere ----
        forced = CONFIG.get("force_partners", {})
        forced = forced.get(key, []) if isinstance(forced, dict) else list(forced)
        forced = [p for p in forced if p not in CPU_PASSES and pass_available(p)]
        f_share = float(CONFIG.get("force_partner_share", 0.0) or 0.0)
        forced_items = 0
        if forced:
            for name in list(plan):
                eff = len(cats[name]) - (1 if gate[name] else 0)
                carry = [p for p in forced
                         if eff <= int(CONFIG.get(p + "_max_width", narrow))]
                if not carry:
                    continue
                share = f_share if f_share > 0 else len(carry) / (len(carry) + 1.0)
                plan[name] = mixed(plan[name], [(p, 1.0 / len(carry)) for p in carry],
                                   share)
                forced_items += 1
        report["forced"] = {"partners": forced, "items": forced_items}

        # ---- does what is left carry the full ensemble? ----
        left = deadline - time.time()
        ratio = ((len(take_all) + len(out_rows))
                 / max(len(rest) + len(val), 1))

        # v38: the lat pass may run the final at its own fixed ensemble size,
        # so taking it does not lower every other pass's E (the slice judged
        # base and lat both at the gate's E and never saw that price)
        lat_E = ((_pc("lat_final_estimators_by_instrument", phase) or {}).get(key) or _pc("lat_final_estimators", phase))

        def plan_need(pl, skip=(), E_=None):
            E_ = E if E_ is None else E_
            counts_ = {}
            for n, parts in pl.items():
                for p, w in parts:
                    if w > 0 and p not in skip:
                        counts_[p] = counts_.get(p, 0) + 1
            return base_cost * ratio * sum(
                float(PASS_COST.get(p, 1.0)) * c / max(len(targets), 1)
                * (min(E_, int(lat_E)) if (p == "lat" and lat_E) else E_) / Eg
                for p, c in counts_.items() if p not in CPU_PASSES)
        need = plan_need(plan)
        if forced and need > left * 0.9:
            # the partners go before the primary's E does
            say("forced partners dropped: %.0fs needed, %.0fs left" % (need, left))
            for name in list(plan):
                plan[name] = [(p, w) for p, w in plan[name] if p not in forced]
                tot_ = sum(w for _, w in plan[name]) or 1.0
                plan[name] = [(p, w / tot_) for p, w in plan[name]]
            report["forced"]["dropped"] = True
            need = plan_need(plan)
        if need > left * 0.9:
            if lat_E:
                E2 = E
                while E2 > 2 and plan_need(plan, E_=E2) > left * 0.9:
                    E2 -= 1
            else:
                E2 = max(2, int(E * (left * 0.9) / need))
            say("final pass: %.0fs needed at E%d, %.0fs left -> E%d"
                % (need, E, left, E2))
            E, Es = E2, E2
        report["final_estimators"] = E

    # ---- the final pass: every pass the plan names, on the items that name it
    t2 = time.time()
    needed = {}
    for n, parts in plan.items():
        for p, w in parts:
            if w > 0:
                needed.setdefault(p, set()).add(n)
    if not needed:
        needed = {"base": set(targets)}
    forced_set = set(report.get("forced", {}).get("partners", []))
    order_f = sorted(needed, key=lambda p: (p != "base", p in forced_set,
                                            -len(needed[p]), p))
    fpasses, fits = {}, 0
    for p in order_f:
        if p in CPU_PASSES:
            if gate_out and p in gate_out:
                fpasses[p] = gate_out[p]
            continue
        only = None if len(needed[p]) >= len(targets) else needed[p]
        if p in forced_set and time.time() >= deadline:
            continue           # every item keeps its plan vector (combine)
        E_p, Es_p = E, Es
        if p == "lat" and ((_pc("lat_final_estimators_by_instrument", phase) or {}).get(key) or _pc("lat_final_estimators", phase)):
            E_p = min(E, int(((_pc("lat_final_estimators_by_instrument", phase) or {}).get(key) or _pc("lat_final_estimators", phase))))
            Es_p = min(Es, int(((_pc("lat_final_estimators_by_instrument", phase) or {}).get(key) or _pc("lat_final_estimators", phase))))
        res, more, _ = run_pass(take_all, out_rows, E_p, Es_p, tag="final",
                                only=only, **pass_kwargs(p))
        fpasses[p] = res
        fits += more
    if gate_out:
        for p in list(fpasses):
            if p in gate_out and p not in CPU_PASSES:
                fpasses[p] = blend(fpasses[p], gate_out[p], float(E),
                                   float(Eg) * float(CONFIG.get("gate_out_weight", 0.0)))
    final = combine(fpasses, plan)
    for name in targets:
        if name not in final:
            final[name] = fpasses[order_f[0]][name]
    report["final_passes"] = {p: len(needed[p]) for p in order_f}
    pooled = {}
    for name, (parts, width, crowd) in final.items():
        pooled[name] = _compose(parts, width, cal, crowd, floor, name in big)
    report.update({"final_seconds": round(time.time() - t2, 1),
                   "final_fits": fits,
                   "call_seconds": round(time.time() - call_started, 1),
                   "deadline_left": round(deadline - time.time(), 1)})
    _REPORT[key] = report
    if debug:
        bad = [n for n, ok in checked if not ok]
        print("[debug] gates %s: %d GIVEN-parent (%d disagreed: %s)"
              % (key, len(checked), len(bad), bad[:4]))
        print("[debug] real_gate %s phase %d: %s" % (key, phase, report))
        print("[debug] real_gate %s: call %.1fs, deadline slack %.1fs, wall %.1fs"
              % (key, time.time() - call_started, deadline - time.time(),
                 time.time() - _STARTED), flush=True)
        print("[debug] terraform %s: factors %s, scaled SSMax calls %d"
              % (key, _TERRA["scale"], _TERRA.get("calls", 0)), flush=True)

    out = []
    order = {name: i for i, name in enumerate(out_rows)}
    if stacked is not None:
        overwrite = {name for name in targets
                     if use_tabpfn and len(cats[name]) <= max_width}
        k = 0
        for row in out_rows:
            i = order[row]
            for j, name in enumerate(targets):
                if not isna[name][row]:
                    continue
                out.append(pooled[name][i] if name in overwrite else stacked[k])
                k += 1
        return out
    for row in out_rows:
        i = order[row]
        for name in items:
            if name in pooled and isna[name][row]:
                out.append(pooled[name][i])
    return out


# ---- v22b: a crash costs the instrument's gains, not the submission --------
_SAFE_OFF = {"gate": False, "unseen_map": False, "tfm_instruments": [],
             "exa_instruments": [], "cau_instruments": [], "force_partners": {},
             "kumo_instruments": [],
             "cpu_instruments": [],
             "stack_instruments": [], "hybrid_instruments": [],
             "pool": False, "recode": False, "delta": False, "nest": False,
             "tfm_preload": False}


def _key_and_phase(frame, schema):
    records = schema["items"]
    targets = [n for n, r in records.items() if r["class"] == "PREDICT"]
    key = ("unhcr" if len(targets) >= 150 else "world_bank"
           if len(targets) >= 40 else "unicef" if len(targets) >= 8 else "other")
    blank = np.zeros(len(frame), bool)
    for name in targets:
        blank |= frame[name].isna().to_numpy()
    split = schema.get("split", {})
    n_fit = int((~blank).sum())
    phase = 2 if (split.get("n_train") and n_fit
                  > split["n_train"] + 0.5 * split.get("n_dev", 0)) else 1
    return key, phase


def _crowd_vectors(frame, schema):
    """The crowd marginals of the visible rows (add-half smoothing), in the
    canonical cell order and option layout the pipeline itself returns."""
    records = schema["items"]
    items = [n for n, r in records.items() if r["class"] in ("GIVEN", "PREDICT")]
    targets = [n for n in items if records[n]["class"] == "PREDICT"]
    sentinel = schema["gated_value"]
    isna, vec = {}, {}
    blank = np.zeros(len(frame), bool)
    for name in targets:
        isna[name] = frame[name].isna().to_numpy()
        blank |= isna[name]
    for name in targets:
        values = list(records[name]["values"]) + (
            [sentinel] if records[name].get("gate") else [])
        seen = frame[name][~blank]
        counts = np.array([float((seen == v).sum()) for v in values]) + 0.5
        vec[name] = counts / counts.sum()
    out = []
    for row in np.flatnonzero(blank):
        for name in items:
            if name in vec and isna[name][row]:
                out.append(vec[name].copy())
    return out


def predict(frame, schema):
    """v22b entry point: the full pipeline; if it raises, the same instrument
    with every optional stage off; if that raises too or the run is nearly
    out of clock, the crowd marginals."""
    try:
        return _predict_full(frame, schema)
    except Exception as exc:
        _REPORT.setdefault("_errors", []).append("full: %r" % (exc,))
        if os.environ.get("SB_DEBUG"):
            print("[debug] full pipeline raised %r; retrying safe" % (exc,),
                  flush=True)
    key, phase = _key_and_phase(frame, schema)
    left = _STARTED + CONFIG["budget"][phase] * CONFIG["safety"] - time.time()
    if left >= float(CONFIG.get("fallback_min_s", 120.0)):
        saved = {k: CONFIG.get(k) for k in _SAFE_OFF}
        CONFIG.update(_SAFE_OFF)
        _SEEN.discard(key)
        try:
            return _predict_full(frame, schema)
        except Exception as exc:
            _REPORT.setdefault("_errors", []).append("safe: %r" % (exc,))
            if os.environ.get("SB_DEBUG"):
                print("[debug] safe pipeline raised %r; crowd" % (exc,),
                      flush=True)
        finally:
            CONFIG.update(saved)
    return _crowd_vectors(frame, schema)
