"""Continued pretraining of NVIDIA Kumo-Tabular (large classifier) on the same
public-survey task stream as cpt2.py (Real-TabPFN recipe: one task per step,
small constant LR). GIVEN codes enter as numerical columns, exactly as the
card feeds them. Tasks with more than 10 classes are skipped (ECOC path).

    python cpt_kumo.py --out runs/kcpt_a --sources r1,...,wgm18 --steps 10000 --lr 1e-6
"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F

import cpt2


def table(X, cols, dev):
    import sdm
    return sdm.TableTensor.from_pandas(df=pd.DataFrame(X, columns=cols),
                                       stypes={c: "numerical" for c in cols}, device=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="", help="safetensors to start from (default: HF large)")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--save", default="2000,5000,10000")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--warm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", default="r8,r10")
    ap.add_argument("--qfrac", type=float, default=0.2, help="query share of each task (0.2 = kd and earlier)")
    ap.add_argument("--boost", default="")
    ap.add_argument("--binned", type=float, default=0.0)
    ap.add_argument("--gated", type=float, default=0.0)
    ap.add_argument("--max_rows", type=int, default=6000)
    ap.add_argument("--rows", default="2000,8001", help="task row range lo,hi")
    ap.add_argument("--model", default="kumo", help="kumo | tabiclv2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import sdm
    import sdm.processing as sp
    from safetensors.torch import load_file, save_file

    cpt2.BINNED = a.binned
    cpt2.ROWS = tuple(int(x) for x in a.rows.split(","))
    rounds = [cpt2.load_source(r, a.gated) for r in a.sources.split(",")]
    pw = np.sqrt([r["n"] for r in rounds])
    for kv in filter(None, a.boost.split(",")):
        k, v = kv.split(":")
        pw = pw * np.array([float(v) if r["name"] == k else 1.0 for r in rounds])
    pw = pw / pw.sum()
    print("source weights", {r["name"]: round(float(p), 3) for r, p in zip(rounds, pw)}, flush=True)

    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    if a.model == "tabiclv2":
        model = sdm.models.TabICLv2(task="classification", device=dev)
    else:
        model = sdm.models.KumoTabular(task="classification", size="large", device=dev)
    inner = next(iter(model.models.values()))
    if a.init:
        inner.load_state_dict({k: v.float() for k, v in load_file(a.init, device="cuda").items()})
    inner.float()
    recipe = model.default_recipe()
    recipe.target = sp.StypeDispatch(categorical=[sp.AlignCategories(),
                                                  sp.ShuffleCategories(method="shift")],
                                     numerical=sp.Standardize())
    recipe.output = sp.Identity()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warm))
    saves = sorted(int(s) for s in a.save.split(","))
    rng = np.random.default_rng(a.seed)

    def save(step):
        sd = {k: v.detach().to(torch.bfloat16).contiguous() if v.is_floating_point() else v
              for k, v in inner.state_dict().items()}
        p = os.path.join(a.out, "step%05d.safetensors" % step)
        save_file(sd, p)
        print("saved", p, flush=True)

    step, losses, skipped, t0 = 0, [], 0, time.time()
    model.train()
    while step < a.steps:
        w = int(rng.integers(7, 18))
        X, y = cpt2.sample_task(rounds[rng.choice(len(rounds), p=pw)], rng, w)
        if len(y) > a.max_rows:
            keep = rng.choice(len(y), a.max_rows, replace=False)
            X, y = X[keep], y[keep]
        n_q = max(30, int(a.qfrac * len(y)))
        perm = rng.permutation(len(y))
        q, c = perm[:n_q], perm[n_q:]
        cls = np.unique(y[c])
        if cls.size < 2 or cls.size > 10 or not np.isin(y[q], cls).all():
            skipped += 1
            continue
        cols = ["g%d" % j for j in range(X.shape[1])]
        yc = sdm.TableTensor.from_pandas(df=pd.DataFrame({"t": y[c].astype(str)}),
                                         stypes={"t": "categorical"}, device=dev)
        yq = sdm.TableTensor.from_pandas(df=pd.DataFrame({"t": y[q].astype(str)}),
                                         stypes={"t": "categorical"}, device=dev)
        opt.zero_grad()
        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x_context=table(X[c], cols, dev), y_context=yc,
                            x_query=table(X[q], cols, dev), recipe=recipe)[0]
            logits, yi = sdm.evaluation.to_class_indices(out, yq, missing_score=-torch.inf)
            loss = F.cross_entropy(logits.float(), yi)
        except (ValueError, RuntimeError) as exc:
            skipped += 1
            if skipped < 5:
                print("skip", repr(exc)[:200], flush=True)
            torch.cuda.empty_cache()
            continue
        if not torch.isfinite(loss):
            skipped += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        losses.append(float(loss.detach()))
        if step % 100 == 0:
            print(json.dumps(dict(step=step, loss100=round(float(np.mean(losses[-100:])), 4),
                                  skipped=skipped, sec=round(time.time() - t0))), flush=True)
        if step in saves:
            save(step)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
