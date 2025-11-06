#!/usr/bin/env python3
"""
Compare train.txt evidences with corpus.txt passages
to measure overlap and consistency.

Usage:
  python check_train_corpus_overlap.py --train data/train.txt --corpus data/corpus.txt --n 200
"""

import json
import re
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List

try:
    import orjson
except Exception:
    orjson = None

def jloads(s: str) -> Any:
    if orjson is not None:
        return orjson.loads(s)
    return json.loads(s)

def clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    return s.lower()

def overlap_ratio(a: str, b: str) -> float:
    """計算兩個字串的重疊比例（以字為單位）。"""
    if not a or not b:
        return 0.0
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200, help="最多取多少筆 train 資料")
    args = ap.parse_args()

    print(f"[INFO] Loading corpus from {args.corpus}")
    corpus_texts = []
    with args.corpus.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = jloads(line)
                txt = obj.get("text", "").strip()
                if txt:
                    corpus_texts.append(clean(txt))
            except Exception:
                continue
    print(f"[INFO] Corpus passages loaded: {len(corpus_texts)}")

    print(f"[INFO] Loading train samples from {args.train}")
    train_evidences = []
    with args.train.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            line = line.strip()
            if not line:
                continue
            obj = jloads(line)
            evs = obj.get("evidences", [])
            labs = obj.get("retrieval_labels", [])
            for ev, lab in zip(evs, labs):
                if lab == 1:
                    train_evidences.append(clean(str(ev)))
    print(f"[INFO] Positive evidences loaded: {len(train_evidences)}")

    # 檢查 overlap
    hits = 0
    examples = []
    for ev in random.sample(train_evidences, min(10, len(train_evidences))):
        best_sim = 0.0
        best_text = None
        for c in corpus_texts:
            if ev in c or c in ev:
                best_sim = 1.0
                best_text = c
                break
            sim = overlap_ratio(ev, c)
            if sim > best_sim:
                best_sim = sim
                best_text = c
        examples.append((best_sim, ev[:120], best_text[:120] if best_text else None))
        if best_sim >= 0.6:
            hits += 1

    hit_rate = hits / len(examples) if examples else 0.0
    print(f"\n[RESULT] Sample overlap hit rate (similarity>=0.6): {hit_rate:.1%}")
    print("[Examples: top 10 evidences]")
    for sim, evfrag, cfrag in examples:
        print(f"  sim={sim:.2f} | EV: {evfrag}")
        if cfrag:
            print(f"       ↳ corpus: {cfrag}")
        else:
            print("       ↳ corpus: <no match>")
        print("-" * 100)

if __name__ == "__main__":
    main()
