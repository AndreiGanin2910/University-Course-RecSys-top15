#!/usr/bin/env python3
"""Local pNDCG@5 checker for DataHakaton 2026 (self-contained).

Usage:
    python metric.py --solution val_solution.csv --submission val_submission.csv

Submission format (P): id,recommendations where recommendations = 5 module ids
separated by spaces, in confidence order, WITHOUT priorities, e.g.:
    s_000123,m_014 m_087 m_031 m_155 m_022
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

# Importance weight of a hit, by the student's true priority of the hit module.
WEIGHTS = {1: 1.00, 2: 0.80, 3: 0.60, 4: 0.45, 5: 0.35}


def _weight(priority: int) -> float:
    if priority in WEIGHTS:
        return WEIGHTS[priority]
    keys = sorted(WEIGHTS)
    return WEIGHTS[keys[0]] if priority < keys[0] else WEIGHTS[keys[-1]]


def pndcg_single(gt: dict, ranked: list):
    if not gt:
        return None
    dcg = 0.0
    for i, module in enumerate(ranked[:5], start=1):
        if module in gt:
            dcg += _weight(gt[module]) / math.log2(i + 1)
    norm = WEIGHTS[1] / math.log2(2)  # one ideal hit = 1.00
    return min(1.0, dcg / norm)


def hit_at_5(gt: dict, ranked: list):
    if not gt:
        return None
    return 1.0 if set(ranked[:5]) & set(gt.keys()) else 0.0


def recall_at_5(gt: dict, ranked: list):
    if not gt:
        return None
    return len(set(ranked[:5]) & set(gt.keys())) / min(5, len(gt))


def parse_ground_truth(raw: str) -> dict:
    out: dict = {}
    text = str(raw).strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    for token in text.split():
        if ":" in token:
            module_id, priority = token.split(":", 1)
            out[module_id] = int(priority)
    return out


def parse_recommendations(raw: str) -> list:
    text = str(raw).strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return [tok.split(":", 1)[0] for tok in text.split() if tok]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Compute pNDCG@5 locally")
    parser.add_argument("--solution", required=True)
    parser.add_argument("--submission", required=True)
    args = parser.parse_args()

    sol: dict = {}
    with Path(args.solution).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sol[row["id"]] = parse_ground_truth(row["ground_truth"])

    sub: dict = {}
    with Path(args.submission).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "recommendations"]:
            print(f"ERROR: expected columns id,recommendations; got {reader.fieldnames}",
                  file=sys.stderr)
            return 1
        for row in reader:
            sub[row["id"]] = parse_recommendations(row["recommendations"])

    missing = sorted(set(sol) - set(sub))
    if missing:
        print(f"ERROR: missing ids in submission: {missing[:5]} ...", file=sys.stderr)
        return 1

    p_scores, h_scores, r_scores = [], [], []
    for sid, gt in sol.items():
        ranked = sub[sid]
        p = pndcg_single(gt, ranked)
        if p is None:
            continue
        p_scores.append(p)
        h_scores.append(hit_at_5(gt, ranked))
        r_scores.append(recall_at_5(gt, ranked))

    n = len(p_scores)
    pndcg = sum(p_scores) / n if n else 0.0
    hit = sum(h_scores) / n if n else 0.0
    rec = sum(r_scores) / n if n else 0.0
    print(f"pNDCG@5 = {pndcg:.4f}   (на {n} студентах)")
    print(f"  справочно: hit@5 = {hit:.4f}   recall@5 = {rec:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
