"""fp32 CPT checkpoint -> bf16 card checkpoint (same format as cpt15000-bf16.ckpt).
    python to_bf16.py runs/cpt_x/step20000.ckpt runs/cpt_x/step20000_bf16.ckpt [BASE]
With BASE, also prints the relative update norm and the bf16-visible change."""
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
a = torch.load(src, map_location="cpu", weights_only=False)
if len(sys.argv) > 3:
    b = torch.load(sys.argv[3], map_location="cpu", weights_only=False)
    num = den = chg = tot = 0
    for k, v in a["state_dict"].items():
        if v.is_floating_point():
            w = b["state_dict"][k]
            num += (v - w).float().norm() ** 2; den += w.float().norm() ** 2
            chg += (v.to(torch.bfloat16) != w.to(torch.bfloat16)).sum().item(); tot += v.numel()
    print("relative update norm %.4f, bf16-visible changed fraction %.3f" % ((num / den) ** .5, chg / tot))
a["state_dict"] = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
                   for k, v in a["state_dict"].items()}
torch.save(a, dst)
print("saved", dst)
