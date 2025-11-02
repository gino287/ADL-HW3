#1) 引入套件
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
import argparse, json, logging, os
from datetime import datetime
from typing import List

from sentence_transformers import LoggingHandler, SentenceTransformer, models
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from datasets import Dataset
# ---- logging ----
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()]
)


#2) 解析參數
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="intfloat/multilingual-e5-small")
parser.add_argument("--train_batch_size", default=192, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--warmup_ratio", default=0.1, type=float)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--data_dir", default="data")
cli = parser.parse_args()
print(cli)

# ---- model ----

# 3)載入嵌入模型
word_embedding_model = models.Transformer(cli.model_name, max_seq_length=cli.max_seq_length)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    normalize_embeddings=True,  # E5 需要 L2 normalize
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])



# 4)讀取資料
# === V3-REWRITE [Data I/O] START ===

# 讀取 corpus.txt，建立 id->text 的對照表
corpus_path = os.path.join(cli.data_dir, "corpus.txt")
corpus = {}
with open(corpus_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        corpus[item["id"]] = item["text"]

logging.info(f"讀取 corpus 完成，共 {len(corpus)} 篇文章")

# 讀取 train.txt，暫存整份資料
train_path  = os.path.join(cli.data_dir, "train.txt")
train_jsonl = []
with open(train_path, "r", encoding="utf-8") as f:
    for line in f:
        train_jsonl.append(json.loads(line))

logging.info(f"讀取 train.txt 完成，共 {len(train_jsonl)} 筆訓練資料")
# === V3-REWRITE [Data I/O] END ===

# 5)建立訓練資料 rows
# === V3-REWRITE [Build rows from train.txt] START ===

def resolve_evidence_to_text(ev: str, corpus_by_id: dict) -> str:
    """
    將 evidence 轉成 passage 文字：
    - 若 ev 剛好是 corpus 的鍵（如 "25749059@5"），就用 corpus 文本。
    - 否則視為已是原文段落（直接回傳）。
    """
    if ev in corpus_by_id:
        return corpus_by_id[ev]
    return ev  # 多數 HW3 的 evidences 本來就放段落全文

def collect_positive_passages(sample: dict, corpus_by_id: dict) -> List[str]:
    """
    從一筆 train 樣本中，收集所有 retrieval_labels==1 的正例段落文字。
    允許多正例；之後可選擇「只取一篇」或「展開成多筆 rows」。
    """
    positives = []
    for ev, lab in zip(sample.get("evidences", []), sample.get("retrieval_labels", [])):
        if lab == 1:
            positives.append(resolve_evidence_to_text(ev, corpus_by_id))
    # 去重（有時候同一段落重複塞進 evidences）
    # 也順便 strip
    dedup = []
    seen = set()
    for p in positives:
        t = p.strip()
        if t and t not in seen:
            dedup.append(t)
            seen.add(t)
    return dedup

rows = []
num_queries_total = 0
num_queries_with_pos = 0
num_rows_generated = 0

for sample in train_jsonl:
    num_queries_total += 1

    # 1) 取 query（E5 前綴）
    #    若你想改用 sample["question"] 也可，但一般建議用 rewrite（較像使用者實際查詢）
    q_raw = (sample.get("rewrite") or sample.get("question") or "").strip()
    if not q_raw:
        # 沒 query 就跳過
        continue
    q = "query: " + q_raw

    # 2) 收集所有正例段落（文字）
    positives = collect_positive_passages(sample, corpus)
    if not positives:
        continue

    num_queries_with_pos += 1

    # 3) 產 row 策略：
    #    A) 「每個 query 只取一篇正例」——訓練穩定、資料均衡（建議 baseline）
    #    B) 「展開成多筆」——對每個正例各產一筆 row（資料較多，但要注意 query 分布）
    # 先採 A 策略：只取第一篇正例
    p = positives[0]
    rows.append({"texts": [q, "passage: " + p]})
    num_rows_generated += 1

    # 若想改用 B 策略，改成以下展開（註解掉 A）：
    # for p in positives:
    #     rows.append({"texts": [q, "passage: " + p]})
    #     num_rows_generated += 1

logging.info(
    f"[Build rows] 總樣本: {num_queries_total}；有正例的 query: {num_queries_with_pos}；產生 rows: {num_rows_generated}"
)

# 4) 檢查 rows 內容，避免空資料繼續往下
if len(rows) == 0:
    raise RuntimeError("[Build rows] 產生的 rows 為 0，請檢查 train.txt / corpus.txt 讀取或標記")

# === V3-REWRITE [Build rows from train.txt] END ===

# 6) 建立 Dataset
# === V3-REWRITE [Step 6 Dataset] START ===

# rows 來自第⑤步，格式為：
# rows = [{"texts": ["query: ...", "passage: ..."]}, ...]

# 建立 Dataset
train_ds_all = Dataset.from_list(rows)
splits = train_ds_all.train_test_split(test_size=0.10, seed=42)  # 9:1
train_ds = splits["train"]
valid_ds = splits["test"]
logging.info(f"[Step 6] Split sizes => train={len(train_ds)} | valid={len(valid_ds)}")
# === V3-REWRITE [Step 6 Dataset] END ===



# 訓練 SentenceTransformer 模型需 dataset、dataloader 及 loss
#7) Trainer 設定與啟動
# === V3-REWRITE [Step 7 DataLoader] START ===
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss

train_args = SentenceTransformerTrainingArguments(
    output_dir="./models/retriever",
    num_train_epochs=cli.epochs,
    per_device_train_batch_size=cli.train_batch_size,
    learning_rate=cli.lr,
    warmup_ratio=cli.warmup_ratio,
    save_strategy="epoch",            # 每 epoch 存檔
    evaluation_strategy="epoch",      # 每 epoch 評估
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    logging_strategy="steps",
    logging_steps=200,
    bf16=True,                        # A100 友善
    fp16=False,
    seed = 42,                   # 設定隨機種子以確保可重現性
    dataloader_num_workers=2, # 設定 DataLoader 的工作緒數
)

loss = MultipleNegativesRankingLoss(model)
trainer = SentenceTransformerTrainer(
    model=model,
    args=train_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    loss=loss,
)
trainer.train()

# 儲存最佳模型（train_args 指到最佳 checkpoint）
model.save_pretrained("./models/retriever")
# === V3-REWRITE [Step 7 DataLoader] END ===


#8) 繪製 loss curve 並儲存模型
import matplotlib.pyplot as plt

state_path = os.path.join(train_args.output_dir, "trainer_state.json")
if os.path.exists(state_path):
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    logs = state.get("log_history", [])

    steps_t, loss_t = [], []
    steps_e, loss_e = [], []
    for log in logs:
        if "loss" in log:
            steps_t.append(log.get("step", None))
            loss_t.append(log["loss"])
        if "eval_loss" in log:
            steps_e.append(log.get("step", None))
            loss_e.append(log["eval_loss"])

    plt.figure()
    if loss_t:
        plt.plot(steps_t, loss_t, label="train_loss")
    if loss_e:
        plt.plot(steps_e, loss_e, label="eval_loss")
    plt.xlabel("steps"); plt.ylabel("loss"); plt.title("Fine-tuning Loss")
    plt.legend(); plt.tight_layout()
    os.makedirs(train_args.output_dir, exist_ok=True)
    plt.savefig(os.path.join(train_args.output_dir, "loss_curve.png"))
else:
    logging.warning("trainer_state.json not found; skip plotting.")