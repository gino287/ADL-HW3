#!/usr/bin/env python3
"""
HW3 JSONL inspector — quick look at the first N lines of train.txt and corpus.txt.

Usage:
  python hw3_inspect_head.py --train /path/to/train.txt --corpus /path/to/corpus.txt --n 20

This prints, for each JSON line:
  - which keys exist
  - short previews of values (strings truncated, arrays summarized, objects show top-level keys)
  - a final "key coverage" summary across the inspected lines
"""

from __future__ import annotations
import textwrap
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:
    import orjson  # type: ignore
except Exception:
    orjson = None  # type: ignore

def jloads(s: str) -> Any:
    if orjson is not None:
        return orjson.loads(s)
    return json.loads(s)

def trunc(s: str, limit: int = 80) -> str:
    s = s.replace("\n", " ").replace("\r", " ")
    if len(s) <= limit:
        return s
    return s[:limit-3] + "..."

def fmt_value(v: Any, limit: int = 80) -> str:
    """Format preview for display, depending on type."""
    if isinstance(v, str):
        return f'"{trunc(v, limit)}"'
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, list):
        if not v:
            return "[]"
        # summarize arrays: length + preview of first element
        first = v[0]
        if isinstance(first, (str, int, float, bool)) or first is None:
            return f"[len={len(v)}; first={fmt_value(first, max(10, limit//2))}]"
        if isinstance(first, Mapping):
            return f"[len={len(v)}; first_obj_keys={list(first.keys())[:6]}]"
        return f"[len={len(v)}; first_type={type(first).__name__}]"
    if isinstance(v, Mapping):
        # show top-level keys
        keys = list(v.keys())
        return "{keys=" + ", ".join(keys[:8]) + ("..." if len(keys) > 8 else "") + "}"
    # fallback
    try:
        return trunc(str(v), limit)
    except Exception:
        return f"<{type(v).__name__}>"

def inspect_jsonl(path: Path, n: int = 20, label: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    records: List[Dict[str, Any]] = []
    coverage: Dict[str, int] = {}
    if not path:
        return records, coverage
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return records, coverage

    print("\n" + "="*80)
    print(f"[INSPECT] {label or path.name} — {path}")
    print("="*80)

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if idx > n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = jloads(line)
            except Exception as e:
                print(f"[{idx:04d}] !! JSON parse error: {e}")
                continue

            if not isinstance(obj, Mapping):
                print(f"[{idx:04d}] !! Not a JSON object (type={type(obj).__name__})")
                continue

            records.append(obj)
            keys = list(obj.keys())
            for k in keys:
                coverage[k] = coverage.get(k, 0) + 1

            # Build a compact preview line
            previews = []
            for k in keys:
                try:
                    v = obj[k]
                    previews.append(f"{k}: {fmt_value(v)}")
                except Exception as e:
                    previews.append(f"{k}: <error {e}>")

            # Print index + the compact preview (wrap long lines for readability)
            head = f"[{idx:04d}] keys={keys}"
            print(head)
            line_text = "  " + " | ".join(previews)
            for seg in textwrap.wrap(line_text, width=140, subsequent_indent="  "):
                print(seg)

    # Summary of key coverage
    if coverage:
        print("\n[SUMMARY] Key coverage across inspected lines:")
        for k, c in sorted(coverage.items(), key=lambda x: (-x[1], x[0])):
            print(f"  - {k}: {c}/{len(records)}")
    else:
        print("\n[SUMMARY] No valid JSON objects parsed.")

    return records, coverage

def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect first N lines of HW3 JSONL files.")
    ap.add_argument("--train", type=Path, help="Path to train.txt (JSONL)")
    ap.add_argument("--corpus", type=Path, help="Path to corpus.txt (JSONL)")
    ap.add_argument("--n", type=int, default=20, help="Number of lines to inspect from each file")
    args = ap.parse_args()

    if args.train is None and args.corpus is None:
        ap.error("Provide at least one of --train or --corpus")

    if args.train:
        inspect_jsonl(args.train, n=args.n, label="train.txt")

    if args.corpus:
        inspect_jsonl(args.corpus, n=args.n, label="corpus.txt")

if __name__ == "__main__":
    main()
