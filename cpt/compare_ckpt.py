"""Compare two TabPFN checkpoints tensor by tensor.

    python compare_ckpt.py runs/cpt_xw_step60000-bf16.ckpt ../weights/cpt_xw_step60000-bf16.ckpt

torch.save output is not byte-deterministic, so a rebuilt checkpoint can have
a different file hash with identical contents. On 2026-11-19, to_bf16.py on
the original step-60,000 checkpoint gave a file whose 595 tensors and
config are identical to the archive file, with a different sha256.
"""
import sys
import torch

a = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
sa, sb = a["state_dict"], b["state_dict"]
diff = [k for k in sa if k not in sb or sa[k].dtype != sb[k].dtype
        or sa[k].shape != sb[k].shape or not torch.equal(sa[k], sb[k])]
diff += [k for k in sb if k not in sa]
meta = [k for k in set(a) | set(b) if k != "state_dict" and repr(a.get(k)) != repr(b.get(k))]
print("tensors: %d vs %d, differing: %d; other fields differing: %s"
      % (len(sa), len(sb), len(diff), meta or "none"))
sys.exit(1 if diff or meta else 0)
