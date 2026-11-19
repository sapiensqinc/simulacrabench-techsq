"""Rebuild the test-phase archive from submission/ and weights/.

    python scripts/build_archive.py          # -> dist/archive/ and dist/Q35n-rebuilt.zip

Every file is checked against MANIFEST.sha256, which lists all 193 files of
the uploaded archive Q35n.zip (sha256 75edc57f...0632abf). The build stops on
any missing, extra or changed file. The rebuilt zip has different zip-level
bytes (timestamps, compression), so the per-file hashes are the identity check.
"""
import hashlib, os, shutil, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC, W = os.path.join(ROOT, "submission"), os.path.join(ROOT, "weights")
OUT = os.path.join(ROOT, "dist", "archive")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


want = {}
with open(os.path.join(ROOT, "MANIFEST.sha256")) as fh:
    for line in fh:
        h, name = line.rstrip("\n").split("  ", 1)
        want[name] = h

shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(SRC, OUT, ignore=shutil.ignore_patterns("__pycache__"))
for name in want:
    if not os.path.exists(os.path.join(OUT, name)) and os.path.exists(os.path.join(W, name)):
        shutil.copy2(os.path.join(W, name), os.path.join(OUT, name))

have = {os.path.relpath(os.path.join(d, f), OUT).replace(os.sep, "/")
        for d, _, fs in os.walk(OUT) for f in fs}
bad = sorted(set(want) ^ have)
bad += sorted(n for n in set(want) & have if sha(os.path.join(OUT, n)) != want[n])
if bad:
    sys.exit("archive does not match MANIFEST.sha256: %s" % ", ".join(bad[:10]))
print("all %d files match MANIFEST.sha256" % len(want))

dst = os.path.join(ROOT, "dist", "Q35n-rebuilt.zip")
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for n in sorted(have):
        z.write(os.path.join(OUT, n), n)
print("wrote", dst)
