#!/usr/bin/env python3
"""Aggregate all-10-task validation runs into SR + Wilson CI + z-test vs sft.

Reads $LOG_ROOT/val2/<label>/task_*_off_*/eval_results.json (output of
validate_all_tasks.sh).

Usage:  python agg_val.py [label1 label2 ...]
        (default: all labels under $LOG_ROOT/val2)
"""
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("LOG_ROOT", "/tmp")) / "val2"


def wilson(s, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = s / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    labels = sys.argv[1:] or sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    print(f"{'label':>8} | {'SR':>6} | {'succ/tot':>9} | {'95% CI':>14} | per-task succ/tot")
    print("-" * 110)
    agg = {}
    for label in labels:
        base = ROOT / label
        per_task = {t: [0, 0] for t in range(10)}
        s = n = 0
        for f in base.glob("task_*_off_*/eval_results.json"):
            d = json.loads(f.read_text())
            for tid, st in d.get("task_stats", {}).items():
                per_task[int(tid)][0] += st["success"]
                per_task[int(tid)][1] += st["total"]
            s += d.get("successes", 0)
            n += d.get("successes", 0) + d.get("failures", 0)
        lo, hi = wilson(s, n)
        agg[label] = (s, n, s / n if n else 0, lo, hi)
        pt = " ".join(f"t{t}:{v[0]}/{v[1]}" for t, v in per_task.items())
        print(
            f"{label:>8} | {100*s/n if n else 0:5.1f}% | {s:>4}/{n:<4} | "
            f"[{100*lo:4.1f},{100*hi:4.1f}]% | {pt}"
        )
    print()
    if "sft" in agg:
        bs, bn = agg["sft"][0], agg["sft"][1]
        bp = bs / bn if bn else 0
        for label in labels:
            if label == "sft":
                continue
            s, n, p, *_ = agg[label]
            if n == 0 or bn == 0:
                continue
            se = math.sqrt(bp * (1 - bp) / bn + p * (1 - p) / n)
            z = (p - bp) / se if se else 0
            print(f"{label} vs sft: Δ={100*(p-bp):+.1f}%  z={z:+.2f}  (|z|>1.96 ~ p<0.05)")


if __name__ == "__main__":
    main()
