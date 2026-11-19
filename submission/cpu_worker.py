"""The CPU members, in their own interpreter.

main.py's gate runs TabPFN passes on the GPU. While it does, this process
fits a different family on the same rows -- gradient-boosted trees and a
logistic regression on one-hot codes with pairwise interactions -- so the
slice can decide, on the delivered rows, whether averaging them in helps.

A separate interpreter rather than a thread: sklearn's OpenMP pool and
torch's are process-wide, a fork after CUDA is initialised is unsafe, and a
thread with the GIL competes with TabPFN's own preprocessing. Here the thread
count is an environment variable set before sklearn is imported, and the
only shared state is one pickle in, one pickle out.

    python cpu_worker.py request.pkl result.pkl

The request carries the codes, the per-item row masks the GPU pass used, the
hyper-parameters and an absolute deadline; the result carries, per kind and
item, the same (kind, ps, pa) | (kind, P) parts main.py composes.
"""
import os
import pickle
import sys
import time

import warnings

import numpy as np

warnings.filterwarnings("ignore")


def _one_hot_design(X, widths, pair_max_levels, min_support):
    """Sparse one-hot of the codes plus one-hot of every pair of columns whose
    widths are both small enough. Columns with fewer than `min_support`
    ones are dropped."""
    import scipy.sparse as sp
    n, d = X.shape
    blocks, offset = [], 0
    Xi = X.astype(np.int64)
    for j in range(d):
        w = int(widths[j]) + 1
        blocks.append(sp.csr_matrix((np.ones(n), (np.arange(n), Xi[:, j])),
                                    shape=(n, w)))
    small = [j for j in range(d) if widths[j] + 1 <= pair_max_levels]
    for a in range(len(small)):
        for b in range(a + 1, len(small)):
            ja, jb = small[a], small[b]
            wb = int(widths[jb]) + 1
            col = Xi[:, ja] * wb + Xi[:, jb]
            w = (int(widths[ja]) + 1) * wb
            blocks.append(sp.csr_matrix((np.ones(n), (np.arange(n), col)),
                                        shape=(n, w)))
    D = sp.hstack(blocks).tocsr()
    support = np.asarray(D.sum(axis=0)).ravel()
    keep = support >= min_support
    return D[:, keep].tocsr()


def _fit(kind, hp, Xtr, ytr, Xq, width, Dtr=None, Dq=None, seed=0):
    """Probabilities over `width` classes on the query rows."""
    classes = np.unique(ytr)
    out = np.zeros((Xq.shape[0], width))
    if len(classes) < 2:
        out[:, int(classes[0])] = 1.0
        return out
    if kind.startswith("donor"):
        # Hot-deck / kNN donors over the GIVEN block. Every other member here
        # is a per-item conditional model fitted independently; this one asks
        # the survey-statistics question instead -- which visible respondents
        # look like this one, and what did they answer -- so it keeps the
        # joint structure the conditional fits smooth away. The answer is a
        # kernel-weighted vote, not a single donor: a single donor is a
        # one-hot vector and a log score punishes that.
        h = dict(hp.get("donor", {}))
        h.update(hp.get("variants", {}).get(kind, {}))
        k = int(h.get("k", 60))
        tau = float(h.get("tau", 1.0))         # exponential kernel width
        prior = float(h.get("prior", 2.0))     # pseudo-counts of the marginal
        wcol = h.get("weights")                # per-column distance weights
        Xi = Xtr.astype(np.int64)
        Xqi = Xq.astype(np.int64)
        d = Xi.shape[1]
        w = np.ones(d) if wcol is None else np.asarray(wcol, float)[:d]
        yk = ytr.astype(np.int64)
        marg = np.bincount(yk, minlength=width).astype(float) + 0.5
        marg = marg / marg.sum()
        k = min(k, len(yk))
        chunk = max(1, int(2e7 // max(len(yk), 1)))
        for lo in range(0, len(Xqi), chunk):
            Q = Xqi[lo:lo + chunk]
            # weighted Hamming: columns that differ add their weight
            dist = ((Q[:, None, :] != Xi[None, :, :]) * w).sum(2)
            part = np.argpartition(dist, k - 1, axis=1)[:, :k]
            dd = np.take_along_axis(dist, part, axis=1)
            ww = np.exp(-dd / max(tau, 1e-6))
            lab = yk[part]
            acc = np.zeros((len(Q), width))
            for c in range(width):
                acc[:, c] = (ww * (lab == c)).sum(1)
            acc += prior * marg[None, :]
            out[lo:lo + chunk] = acc / acc.sum(1, keepdims=True)
        return out
    if kind.startswith("hgb"):
        from sklearn.ensemble import HistGradientBoostingClassifier
        h = dict(hp.get("hgb", {}))
        h.update(hp.get("variants", {}).get(kind, {}))
        clf = HistGradientBoostingClassifier(
            max_iter=int(h.get("max_iter", 120)),
            learning_rate=float(h.get("learning_rate", 0.08)),
            max_leaf_nodes=int(h.get("max_leaf_nodes", 15)),
            min_samples_leaf=int(h.get("min_samples_leaf", 20)),
            l2_regularization=float(h.get("l2_regularization", 1.0)),
            categorical_features=list(range(Xtr.shape[1])),
            early_stopping=False, random_state=seed)
        clf.fit(Xtr, ytr)
        pr = clf.predict_proba(Xq)
        out[:, np.asarray(clf.classes_, np.int64)] = pr
    elif kind.startswith("lr2"):
        from sklearn.linear_model import LogisticRegression
        C = float(hp.get("variants", {}).get(kind, {}).get("C", hp.get("lr2_C", 0.5)))
        clf = LogisticRegression(C=C, max_iter=400, tol=1e-3)
        clf.fit(Dtr, ytr)
        pr = clf.predict_proba(Dq)
        out[:, np.asarray(clf.classes_, np.int64)] = pr
    elif kind.startswith("tab"):
        # Empirical-Bayes backoff over the two GIVEN columns that carry the
        # most (bias-corrected) information: P(y|a,b) shrunk to P(y|a)
        # shrunk to the marginal, prior weight `m` at each level.
        m = float(hp.get("variants", {}).get(kind, {}).get("m", 8.0))
        Xi = Xtr.astype(np.int64)
        Xqi = Xq.astype(np.int64)
        d = Xi.shape[1]
        widths_ = np.maximum(Xi.max(axis=0), Xqi.max(axis=0)) + 1
        yk = ytr.astype(np.int64)
        marg = np.bincount(yk, minlength=width).astype(float) + 0.5
        marg /= marg.sum()

        def counts_of(cols):
            idx = np.zeros(len(yk), np.int64)
            for c in cols:
                idx = idx * int(widths_[c]) + Xi[:, c]
            ncell = int(np.prod([int(widths_[c]) for c in cols]))
            cnt = np.zeros((ncell, width))
            np.add.at(cnt, (idx, yk), 1.0)
            return cnt

        def mi_of(cols):
            cnt = counts_of(cols)
            joint = cnt / cnt.sum()
            px = joint.sum(1, keepdims=True)
            py = joint.sum(0, keepdims=True)
            nz = joint > 0
            mi = float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())
            kx, ky = int((px > 0).sum()), int((py > 0).sum())
            return mi - (int(nz.sum()) - kx - ky + 1) / (2.0 * len(yk))

        first = max(range(d), key=lambda c: mi_of([c]))
        rest_ = [c for c in range(d) if c != first]
        second = max(rest_, key=lambda c: mi_of([first, c])) if rest_ else None
        # level 1: P(y | a)
        c1 = counts_of([first])
        p1 = (c1 + m * marg[None, :]) / (c1.sum(1, keepdims=True) + m)
        if second is None:
            out[:] = p1[Xqi[:, first]]
        else:
            w2 = int(widths_[second])
            c2 = counts_of([first, second])          # rows indexed a*w2+b
            prior2 = np.repeat(p1, w2, axis=0)      # cell (a,b) -> p1[a]
            p2 = (c2 + m * prior2) / (c2.sum(1, keepdims=True) + m)
            out[:] = p2[Xqi[:, first] * w2 + Xqi[:, second]]
    else:
        raise ValueError(kind)
    return out


def run(req):
    X = req["X"]                    # (n, d) float codes, unseen = width
    widths = req["widths"]          # per GIVEN column, number of levels
    take, query = req["take"], req["query"]
    items = req["items"]            # list of dicts: name, width, gated, y (codes on take), keep (bool on take)
    kinds = list(req["kinds"])
    hp = req.get("hp", {})
    deadline = float(req["deadline"])
    smoothing = float(hp.get("smoothing", 0.5))
    seed = int(hp.get("seed", 0))
    D = None
    if any(k.startswith("lr2") for k in kinds):
        D = _one_hot_design(X, widths, int(hp.get("lr2_pair_max_levels", 12)),
                            int(hp.get("lr2_min_support", 5)))
    Xtr_all, Xq = X[take], X[query]
    Dtr_all = D[take] if D is not None else None
    Dq = D[query] if D is not None else None
    result = {k: {} for k in kinds}
    timing = {k: 0.0 for k in kinds}
    done = {k: 0 for k in kinds}
    for kind in kinds:
        t0 = time.time()
        for it in items:
            if time.time() > deadline:
                break
            name, width, gated = it["name"], int(it["width"]), bool(it["gated"])
            y_all, keep = it["y"], it["keep"]
            counts = np.bincount(y_all[keep], minlength=width).astype(
                np.float64) + smoothing
            crowd = counts / counts.sum()
            try:
                if gated:
                    is_sent = (y_all == width - 1).astype(np.int64)
                    asked = keep & (y_all < width - 1)
                    if is_sent[keep].min() == is_sent[keep].max():
                        ps = np.full(len(query), float(is_sent[keep].mean()))
                    else:
                        ps = _fit(kind, hp, Xtr_all[keep], is_sent[keep], Xq, 2,
                                  None if D is None else Dtr_all[keep], Dq,
                                  seed)[:, 1]
                    if asked.sum() >= 32 and np.unique(y_all[asked]).size >= 2:
                        pa = _fit(kind, hp, Xtr_all[asked], y_all[asked], Xq,
                                  width - 1,
                                  None if D is None else Dtr_all[asked], Dq,
                                  seed)
                    else:
                        c2 = np.bincount(y_all[asked], minlength=width - 1) \
                            + smoothing
                        pa = np.tile(c2 / c2.sum(), (len(query), 1))
                    ps = np.clip(ps, 1e-6, 1 - 1e-6)
                    pa = pa / np.maximum(pa.sum(1, keepdims=True), 1e-12)
                    result[kind][name] = ("hurdle", ps, pa, width, crowd)
                elif keep.sum() >= 32 and np.unique(y_all[keep]).size >= 2:
                    P = _fit(kind, hp, Xtr_all[keep], y_all[keep], Xq, width,
                             None if D is None else Dtr_all[keep], Dq, seed)
                    P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
                    result[kind][name] = ("full", P, width, crowd)
                else:
                    continue
                done[kind] += 1
            except Exception as exc:  # one item never sinks the pass
                result.setdefault("_errors", []).append((kind, name, repr(exc)))
        timing[kind] = round(time.time() - t0, 1)
    return {"result": result, "timing": timing, "done": done,
            "design_cols": 0 if D is None else int(D.shape[1])}


def main():
    req_path, out_path = sys.argv[1], sys.argv[2]
    with open(req_path, "rb") as fh:
        req = pickle.load(fh)
    out = run(req)
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out_path)


if __name__ == "__main__":
    main()
