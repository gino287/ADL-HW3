#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_train_stats.py — 只做偵測/統計，不修改任何訓練或推論程式。
用法：
  python tools/scan_train_stats.py \
    --train_path data/train.txt \
    --ce_model cross-encoder/ms-marco-MiniLM-L-12-v2 \
    --limit 200 \
    --csv_out work/train_scan_sample.csv
"""
import argparse, json, math, sys, os
from collections import Counter, defaultdict
from statistics import mean
from typing import List, Dict, Any, Tuple, Optional

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, default="data/train.txt")
    p.add_argument("--ce_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    p.add_argument("--limit", type=int, default=200, help="只掃前 N 行；-1 表示全量")
    p.add_argument("--csv_out", type=str, default="", help="可選：把逐行統計輸出成 CSV")
    return p.parse_args()

def safe_get(dic: Dict[str, Any], key: str, default=None):
    v = dic.get(key, default)
    return v

def load_tokenizer(model_name: str):
    # CrossEncoder 也用 HF tokenizer；不建模、只量 token 長度
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    return tok

def tokens_len_pair(tok, q: str, p: str) -> int:
    # 不截斷，純量測 pair 長度（含 special tokens）
    enc = tok(q, p, add_special_tokens=True, truncation=False, return_attention_mask=False)
    return len(enc["input_ids"])

def pct(x: List[float], q: float) -> float:
    if not x: return float("nan")
    xs = sorted(x)
    k = (len(xs)-1)*q
    f = math.floor(k); c = math.ceil(k)
    if f == c: return float(xs[int(k)])
    return xs[f] + (xs[c]-xs[f])*(k-f)

def main():
    args = parse_args()
    tok = load_tokenizer(args.ce_model)

    total = 0
    bad_mismatch = 0
    has_query_prefix = 0

    # 逐 qid 彙整
    qid_pos_cnt = Counter()
    qid_neg_cnt = Counter()
    qid_dup_ev   = Counter()  # 是否有重複 evidence
    qid_len_stats: Dict[str, List[int]] = defaultdict(list)  # 每 qid 收集 pair token 長度

    # 逐行 CSV 匯出（可選）
    rows_for_csv = []

    with open(args.train_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if args.limit != -1 and line_idx > args.limit:
                break
            line = line.strip()
            if not line: continue
            obj = json.loads(line)

            qid = str(safe_get(obj, "qid", f"row#{line_idx}"))
            q = safe_get(obj, "rewrite", "") or safe_get(obj, "question", "") or ""
            evs: List[str] = safe_get(obj, "evidences", []) or []
            labs: List[Any] = safe_get(obj, "retrieval_labels", []) or []

            if q.strip().lower().startswith("query:"):
                has_query_prefix += 1

            # 清點標籤
            if len(evs) != len(labs):
                bad_mismatch += 1

            pos_c = 0
            neg_c = 0
            for lab in labs:
                try:
                    v = int(lab)
                except:
                    try:
                        v = int(round(float(lab)))
                    except:
                        continue
                if v == 1: pos_c += 1
                elif v == 0: neg_c += 1

            qid_pos_cnt[qid] += pos_c
            qid_neg_cnt[qid] += neg_c

            # 重複 evidence 檢查（以純文字比對）
            seen = set()
            dup = False
            for e in evs:
                if not isinstance(e, str): continue
                t = e.strip()
                if t in seen:
                    dup = True
                else:
                    seen.add(t)
            if dup: qid_dup_ev[qid] += 1

            # 量測 CE pair token 長度（每個 evidence 都量）
            pair_lens = []
            for e in evs:
                if not isinstance(e, str): continue
                try:
                    L = tokens_len_pair(tok, q, e)
                    pair_lens.append(L)
                    qid_len_stats[qid].append(L)
                except Exception:
                    pass

            total += 1

            # 逐行輸出（可選）
            if args.csv_out:
                row = {
                    "idx": line_idx,
                    "qid": qid,
                    "pos_in_row": pos_c,
                    "neg_in_row": neg_c,
                    "has_dup_evidence": int(dup),
                    "query_has_prefix_query": int(q.strip().lower().startswith("query:")),
                    "pair_len_min": min(pair_lens) if pair_lens else "",
                    "pair_len_p50": pct(pair_lens, 0.50) if pair_lens else "",
                    "pair_len_p90": pct(pair_lens, 0.90) if pair_lens else "",
                    "pair_len_max": max(pair_lens) if pair_lens else "",
                }
                rows_for_csv.append(row)

    # 匯總
    qids = set(qid_pos_cnt.keys()) | set(qid_neg_cnt.keys())
    multi_pos_qids = [q for q in qids if qid_pos_cnt[q] >= 2]
    zero_pos_qids  = [q for q in qids if qid_pos_cnt[q] == 0]

    all_pair_lens = [L for q in qid_len_stats for L in qid_len_stats[q]]

    print("=== scan_train_stats summary ===")
    print(f"scanned_lines         : {total}")
    print(f"mismatch_evs_labels   : {bad_mismatch}")
    print(f"qids_seen             : {len(qids)}")
    print(f"qids_with_multi_pos   : {len(multi_pos_qids)}")
    print(f"qids_with_zero_pos    : {len(zero_pos_qids)}")
    print(f"lines_query_has_prefix: {has_query_prefix}")

    if all_pair_lens:
        print("pair_token_length  (no truncation):")
        print(f"  min={min(all_pair_lens)}  p50={int(pct(all_pair_lens,0.5))}  p90={int(pct(all_pair_lens,0.9))}  max={max(all_pair_lens)}")
    else:
        print("pair_token_length: (no data)")

    # 若要 CSV
    if args.csv_out:
        import csv, pathlib
        pathlib.Path(os.path.dirname(args.csv_out) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.csv_out, "w", newline="", encoding="utf-8") as wf:
            w = csv.DictWriter(wf, fieldnames=list(rows_for_csv[0].keys()) if rows_for_csv else
                               ["idx","qid","pos_in_row","neg_in_row","has_dup_evidence","query_has_prefix_query",
                                "pair_len_min","pair_len_p50","pair_len_p90","pair_len_max"])
            w.writeheader()
            for r in rows_for_csv:
                w.writerow(r)
        print(f"[saved] CSV -> {args.csv_out}")

if __name__ == "__main__":
    main()
