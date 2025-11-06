#!/usr/bin/env python3
"""
HW3 QA preview — show first N examples of question / rewrite / answer
from train.txt (JSONL).

Usage:
  python hw3_preview_qa.py --train /path/to/train.txt --n 20
"""

import argparse, json
from pathlib import Path
from typing import Any, Mapping

try:
    import orjson  # type: ignore
except Exception:
    orjson = None  # type: ignore

def jloads(s: str) -> Any:
    if orjson is not None:
        return orjson.loads(s)
    return json.loads(s)

def trunc(s: str, limit: int = 200) -> str:
    s = s.replace("\n", " ").replace("\r", " ")
    if len(s) <= limit:
        return s
    return s[:limit - 3] + "..."

def preview_train(path: Path, n: int = 20):
    if not path.exists():
        raise FileNotFoundError(path)
    print("=" * 80)
    print(f"[INSPECT] {path.name} — first {n} lines (question / rewrite / answer)")
    print("=" * 80)

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i > n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj: Mapping[str, Any] = jloads(line)
            except Exception as e:
                print(f"[{i:04d}] !! JSON error: {e}")
                continue

            q = obj.get("question", "")
            rw = obj.get("rewrite", "")
            ans = ""
            # answer might be inside dict like {"text": "..."} or plain string
            raw_ans = obj.get("answer")
            if isinstance(raw_ans, Mapping):
                ans = raw_ans.get("text", "")
            elif isinstance(raw_ans, str):
                ans = raw_ans
            else:
                ans = str(raw_ans) if raw_ans is not None else ""

            print(f"[{i:04d}]")
            print(f"Q : {trunc(q)}")
            print(f"RW: {trunc(rw)}")
            print(f"AN: {trunc(ans)}")
            print("-" * 80)

def main():
    ap = argparse.ArgumentParser(description="Preview question/rewrite/answer from train.txt")
    ap.add_argument("--train", type=Path, required=True, help="Path to train.txt")
    ap.add_argument("--n", type=int, default=20, help="Number of lines to show")
    args = ap.parse_args()
    preview_train(args.train, n=args.n)

if __name__ == "__main__":
    main()
