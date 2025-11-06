# tools/mine_hard_negatives.py
import json, sqlite3, faiss, argparse, os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_index(ix_path):
    return faiss.read_index(ix_path)

def load_sqlite(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    return conn, cur

def get_text_by_ids(cur, ids):
    # rowid 從 0 起，SQLite rowid 從 0 插入；用 =? 直接取
    res = []
    for _id in ids:
        cur.execute("SELECT text FROM passages WHERE rowid = ?", (int(_id),))
        row = cur.fetchone()
        res.append(row[0] if row else "")
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data/train.txt")
    ap.add_argument("--index_path", default="vector_database/passage_index.faiss")
    ap.add_argument("--sqlite_path", default="vector_database/passage_store.db")
    ap.add_argument("--retriever_model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--per_q_hard", type=int, default=2)
    ap.add_argument("--out_path", default="data/hardneg.jsonl")
    args = ap.parse_args()

    ix = load_index(args.index_path)
    conn, cur = load_sqlite(args.sqlite_path)
    model = SentenceTransformer(args.retriever_model, device="cuda")
    out = open(args.out_path, "w", encoding="utf-8")

    with open(args.train_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="mining hard negatives"):
            if not line.strip(): continue
            obj = json.loads(line)
            qid = obj["qid"]
            q = obj.get("rewrite") or obj.get("question") or ""
            evs = obj.get("evidences") or []
            labs = obj.get("retrieval_labels") or []
            # gold 文字 + 原始 4 個負例文字
            gold = set([evs[i].strip() for i,lb in enumerate(labs) if int(lb)==1])
            easy = set([evs[i].strip() for i,lb in enumerate(labs) if int(lb)==0])

            # e5 規範：查詢要有 "query:" 前綴；且 normalize_embeddings=True 才對齊建庫
            q_vec = model.encode([f"query: {q}"], normalize_embeddings=True)
            D, I = ix.search(q_vec, args.topk)   # 內積相似

            cand_texts = get_text_by_ids(cur, I[0])
            hard = []
            for t in cand_texts:
                t_ = (t or "").strip()
                if not t_: continue
                if t_ in gold:    # 排除正例
                    continue
                if t_ in easy:    # 排除已存在的 4 個負例
                    continue
                hard.append(t_)
                if len(hard) >= args.per_q_hard:
                    break

            out.write(json.dumps({"qid": qid, "hard_negatives": hard}, ensure_ascii=False) + "\n")

    out.close()
    conn.close()

if __name__ == "__main__":
    main()

# # 先確保你已建立 passage 向量庫（做過就不用重跑）
# python save_embeddings.py \
#   --data_folder ./data \
#   --file_name corpus.txt \
#   --output_folder ./vector_database \
#   --build_db \
#   --retriever_model_path ./models/retriever

# # 接著抽 hard negatives（每 qid 抽 2 條）
# python tools/mine_hard_negatives.py \
#   --train_path data/train.txt \
#   --index_path vector_database/passage_index.faiss \
#   --sqlite_path vector_database/passage_store.db \
#   --retriever_model ./models/retriever \
#   --topk 50 \
#   --per_q_hard 2 \
#   --out_path data/hardneg.jsonl
