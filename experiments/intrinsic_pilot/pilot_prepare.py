"""Intrinsic-dimension pilot (2026-10-11): aligned distinct-string matrices of the branches10
phrase units (run from the repo root in langembed-ml; see docs/intrinsic-dimension.md, section 8).
mpnet vectors: scripts/embed_article_units_cpu.py on strings.jsonl."""

import json
from pathlib import Path

import numpy as np

SRC = Path("experiments/branches10")
OUT = Path("experiments/intrinsic_pilot")
(OUT / "emb").mkdir(parents=True, exist_ok=True)
with open(SRC / "input/units.jsonl", encoding="utf-8") as f:
    texts = [json.loads(line)["text"] for line in f if line.strip()]
u2s = np.load(SRC / "input/unit_to_string.npy")
n_str = int(u2s.max()) + 1
first = np.full(n_str, -1, dtype=np.int64)
for row in range(len(u2s) - 1, -1, -1):
    first[u2s[row]] = row
assert (first >= 0).all()
with open(OUT / "strings.jsonl", "w", encoding="utf-8") as f:
    for s, row in enumerate(first):
        f.write(json.dumps({"string_id": s, "text": texts[row]}, ensure_ascii=False) + "\n")
for name in ["A0", "A", "B", "C", "CBOW"]:
    v = np.load(SRC / f"emb/{name}_vecs.npy", mmap_mode="r")
    # branches10 stores one row per distinct string (string id = row); per-unit files are mapped
    m = np.asarray(v if v.shape[0] == n_str else v[first], dtype=np.float32)
    np.save(OUT / f"emb/{name}.npy", m)
    print(name, v.shape, "->", m.shape)
print(n_str, "distinct strings")
