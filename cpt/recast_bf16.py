"""Recast a TabPFN safetensors checkpoint to bf16, header metadata preserved (as v23's v35h).

    python recast_bf16.py <in.safetensors> <out-bf16.safetensors>
"""
import sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
with safe_open(src, framework="pt", device="cpu") as f:
    meta = f.metadata() or {}
    tensors = {k: f.get_tensor(k) for k in f.keys()}
n = sum(t.numel() for t in tensors.values() if t.is_floating_point())
out = {k: (t.to(torch.bfloat16) if t.is_floating_point() else t).contiguous() for k, t in tensors.items()}
save_file(out, dst, metadata=meta)
print("recast %d floating-point values of %d tensors -> %s" % (n, len(out), dst))
