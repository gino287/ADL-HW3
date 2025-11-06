#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, random, re, sqlite3, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except Exception:
    faiss = None
    SentenceTransformer = None

BR_TAG_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")

def clean_text(t: str) -> str:
    t = BR_TAG_RE.sub(" ", t or "")
    t = SPACE_RE.sub(" ", t)
    return t.strip()

def preprocess_text(t: Optional[str]) -> Optional[str]:
    if t is None: return None
    t = clean_text(t)
    if len(t) < 3: return None
    if len(t) > 10000: return None
    return t

def load_train(train_path: Path) -> Dict[str, Dict[str, object]]:
    mp: Dict[str, Dict[str, object]] = {}
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            o = json.loads(line)
            qid = str(o.get("qid") or "")
            if not qid: continue
            q = preprocess_text(o.get("rewrite") or o.get("question") or "")
            if q is None: continue
            evs = o.get("evidences") or []
            labs = o.get("retrieval_labels") or []
            if not evs or not labs or len(evs) != len(labs): continue
            gold, easy = None, set()
            for ev, lb in zip(evs, labs):
                if not isinstance(ev, str): continue
                evc = preprocess_text(ev)
                if evc is None: continue
                try: lbv = int(lb)
                except Exception:
                    try: lbv = int(round(float(lb)))
                    except Exception: continue
                if lbv == 1 and gold is None: gold = evc
                elif lbv == 0: easy.add(evc)
            if gold: mp[qid] = {"query": q, "gold": gold, "easy": easy}
    return mp

def load_hard(hard_path: Path) -> Dict[str, List[str]]:
    mp: Dict[str, List[str]] = {}
    with hard_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            o = json.loads(line)
            qid = str(o.get("qid") or "")
            if not qid: continue
            arr = []
            for t in (o.get("hard_negatives") or []):
                tt = preprocess_text(t)
                if tt: arr.append(tt)
            mp[qid] = arr
    return mp

def open_faiss(ix_path: Optional[Path]):
    if ix_path is None or not ix_path.exists() or faiss is None: return None
    return faiss.read_index(str(ix_path))

def open_sqlite(db_path: Optional[Path]):
    if db_path is None or not db_path.exists(): return None
    conn = sqlite3.connect(str(db_path))
    return conn, conn.cursor()

def get_texts_by_rowids(cur, ids: List[int]) -> List[str]:
    out = []
    for rid in ids:
        cur.execute("SELECT text FROM passages WHERE rowid = ?", (int(rid),))
        row = cur.fetchone()
        out.append(row[0] if row else "")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data/train.txt")
    ap.add_argument("--hardneg_path", default="data/hardneg.jsonl")
    ap.add_argument("--k", type=int, default=12, help="抽查幾個 qid")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk_check", action="store_true", help="啟用 FAISS Top-K 檢查")
    ap.add_argument("--index_path", default="vector_database/passage_index.faiss")
    ap.add_argument("--sqlite_path", default="vector_database/passage_store.db")
    ap.add_argument("--retriever_model", default="./models/retriever")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    train = load_train(Path(args.train_path))
    hard = load_hard(Path(args.hardneg_path))
    qids = list(set(train.keys()) & set(hard.keys()))
    if not qids:
        print("沒有可抽查的交集 qid，請確認 train.txt 與 hardneg.jsonl。")
        sys.exit(1)

    random.seed(args.seed)
    random.shuffle(qids)
    pick = qids[: min(args.k, len(qids))]

    # 可選 Top-K 檢查
    idx = cur = model = None
    conn = None
    if args.topk_check:
        if SentenceTransformer is None or faiss is None:
            print("缺 sentence_transformers 或 faiss，無法做 Top-K 檢查。", file=sys.stderr)
            args.topk_check = False
        else:
            idx = open_faiss(Path(args.index_path))
            tmp = open_sqlite(Path(args.sqlite_path))
            if not idx or not tmp:
                print("找不到 index 或 passage_store.db，Top-K 檢查關閉。", file=sys.stderr)
                args.topk_check = False
            else:
                conn, cur = tmp
                model = SentenceTransformer(args.retriever_model, device="cuda")

    def enc_query(q: str):
        return model.encode([f"query: {q}"], normalize_embeddings=True)

    print(f"\n=== spot-check {len(pick)} qids ===")
    for qi, qid in enumerate(pick, 1):
        info = train[qid]
        q = info["query"]  # type: ignore
        gold = info["gold"]  # type: ignore
        easy = info["easy"]  # type: ignore
        hlist = hard[qid]

        print(f"\n[{qi}/{len(pick)}] qid={qid}")
        print(f"Q : {q[:120]}{'...' if len(q)>120 else ''}")
        print(f"G : {gold[:120]}{'...' if len(gold)>120 else ''}")
        print(f"easy_neg (#{len(easy)}): sample -> {list(easy)[:2]}")
        print(f"hard_neg (#{len(hlist)}):")

        rank_map = {}
        if args.topk_check:
            D, I = idx.search(enc_query(q), args.topk)  # type: ignore
            texts = get_texts_by_rowids(cur, list(I[0]))  # type: ignore
            for r, t in enumerate(texts, start=1):
                tt = preprocess_text(t)
                if tt: rank_map[tt] = r

        seen = set()
        for t in hlist:
            tag = []
            if t == gold: tag.append("DUP_GOLD")
            if t in easy: tag.append("DUP_EASY")
            if t in seen: tag.append("DUP_REPEAT")
            seen.add(t)
            if args.topk_check:
                r = rank_map.get(t, -1)
                tag.append(f"RANK={r}" if r != -1 else "NOT_IN_TOPK")
            tag_s = "[" + ",".join(tag) + "]" if tag else "[OK]"
            print(f"  - {tag_s} {t[:120]}{'...' if len(t)>120 else ''}")

    if conn: conn.close()
    print("\n完成。")
if __name__ == "__main__":
    main()
