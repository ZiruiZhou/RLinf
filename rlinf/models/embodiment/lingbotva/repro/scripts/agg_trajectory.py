#!/usr/bin/env python3
"""Aggregate deterministic per-task eval_results.json into an SR trajectory.

Reads $LOG_ROOT/valt26/<label>/task_<t>_off_<off>/eval_results.json (the output
of eval_tasks.sh) and prints, per label, the success rate with a Wilson 95% CI,
the per-task breakdown, and a two-proportion z-test vs the `sft` label.

Usage:  python agg_trajectory.py [label1 label2 ...]
        (default: all labels under $LOG_ROOT/valt26, in the canonical order)
"""
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("LOG_ROOT", "/tmp")) / "valt26"
ORDER = ["sft", "exact10", "exact15", "exact20", "exact25", "exact30"]
TASKS = (2, 6)


def wilson(s, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = s / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load(label):
    base = ROOT / label
    s = n = 0
    per_task = {t: [0, 0] for t in TASKS}
    for f in base.glob("task_*_off_*/eval_results.json"):
        d = json.loads(f.read_text())
        for tid, st in d.get("task_stats", {}).items():
            tid = int(tid)
            if tid in per_task:
                per_task[tid][0] += st["success"]
                per_task[tid][1] += st["total"]
        s += d.get("successes", 0)
        n += d.get("successes", 0) + d.get("failures", 0)
    return s, n, per_task


def main():
    labels = sys.argv[1:] or [x for x in ORDER if (ROOT / x).exists()]
    agg = {lab: load(lab) for lab in labels if (ROOT / lab).exists()}
    bs, bn, _ = agg.get("sft", (0, 0, None))
    bp = bs / bn if bn else 0
    print(f"=== EXACT-GRADIENT TRAJECTORY (deterministic, tasks {TASKS}) ===")
    for lab in labels:
        if lab not in agg:
            continue
        s, n, t = agg[lab]
        lo, hi = wilson(s, n)
        p = s / n if n else 0
        extra = ""
        if lab != "sft" and bn and n:
            se = math.sqrt(bp * (1 - bp) / bn + p * (1 - p) / n)
            z = (p - bp) / se if se else 0
            extra = f"  | vs sft Δ={100*(p-bp):+.1f}% z={z:+.2f}"
        pt = " ".join(f"t{k}:{v[0]}/{v[1]}" for k, v in t.items())
        print(
            f"{lab:>8}: {100*p:5.1f}% ({s}/{n}) "
            f"CI[{100*lo:.0f},{100*hi:.0f}] | {pt}{extra}"
        )


if __name__ == "__main__":
    main()
