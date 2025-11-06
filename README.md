# ADL HW3 - Retrieval-Augmented Generation (RAG)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.56.1-yellow.svg)](https://huggingface.co/transformers/)

本專案實作一個完整的 RAG 系統，包含 **Retriever**（雙塔模型）、**Reranker**（Cross-Encoder）和 **Generator**（LLM）三個階段，用於問答任務。

---

## 📋 目錄

- [專案架構](#專案架構)
- [環境設置](#環境設置)
- [快速開始](#快速開始)
- [完整訓練流程](#完整訓練流程)
- [推論與評估](#推論與評估)
- [工具說明](#工具說明)
- [實驗重現](#實驗重現)
- [目錄結構](#目錄結構)

---

## 🏗️ 專案架構

```
RAG Pipeline: Query → Retriever → Reranker → Generator → Answer

1. Retriever (Bi-Encoder): 從 corpus 中檢索 Top-K 相關段落
2. Reranker (Cross-Encoder): 重新排序，選出 Top-M 最相關段落
3. Generator (LLM): 基於檢索到的段落生成答案
```

---

## 🛠️ 環境設置

### 系統需求
- **Python**: 3.12
- **CUDA**: 12.4 (for GPU support)
- **GPU**: 建議使用 A100 (至少 16GB VRAM)

### 安裝步驟

#### 1. Clone 專案
```bash
git clone https://github.com/gino287/ADL-HW3.git
cd ADL-HW3
```

#### 2. 建立虛擬環境（建議）
```bash
python3.12 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

#### 3. 安裝套件
```bash
pip install -r requirements.txt
```

**requirements.txt 內容：**
- `transformers==4.56.1` - HuggingFace 模型庫
- `torch==2.8.0` - PyTorch (CUDA 12.4)
- `datasets==4.0.0` - 資料集處理
- `sentence-transformers==5.1.0` - 雙塔模型與 Cross-Encoder
- `faiss-gpu-cu12==1.12.0` - 向量檢索引擎
- `python-dotenv==1.1.1` - 環境變數管理
- `accelerate==1.10.1` - 分散式訓練加速
- `gdown` - Google Drive 下載工具

#### 4. 設定 HuggingFace Token
建立 `.env` 檔案並加入您的 HuggingFace token：
```bash
echo 'hf_token="your_huggingface_token_here"' > .env
```
> 取得 token: https://huggingface.co/settings/tokens

---

## 🚀 快速開始

### 1. 下載預訓練模型
```bash
bash download.sh
```
這會下載已訓練好的 Retriever 和 Reranker 模型到 `models/` 目錄。

### 2. 建立向量資料庫
```bash
python code/data_preparation/save_embeddings.py \
  --retriever_model_path ./models/retriever \
  --build_db
```

### 3. 執行推論
```bash
python code/evaluation/inference_batch.py \
  --retriever_model_path ./models/retriever \
  --reranker_model_path ./models/reranker \
  --test_data_path ./data/test_open.txt
```

輸出檔案：`result.json`

---

## 🎓 完整訓練流程

### Step 1: 資料準備與檢查

#### 1.1 檢查資料結構
```bash
# 查看 JSONL 檔案結構
python code/check_tool/hw3_inspect_head.py \
  --train data/train.txt \
  --corpus data/corpus.txt \
  --n 20

# 預覽問答對內容
python code/check_tool/hw3_preview_qa.py \
  --train data/train.txt \
  --n 20
```

#### 1.2 統計資料特性
```bash
# 掃描訓練資料統計（token 長度、標籤分布等）
python code/evaluation/scan_train_stats.py \
  --train_path data/train.txt \
  --ce_model cross-encoder/ms-marco-MiniLM-L-12-v2 \
  --limit 200 \
  --csv_out work/train_stats.csv
```

#### 1.3 檢查訓練資料與語料庫重疊
```bash
python code/check_tool/check_train_corpus_overlap.py \
  --train data/train.txt \
  --corpus data/corpus.txt \
  --n 200
```

---

### Step 2: 挖掘 Hard Negatives

Hard negatives 是訓練 Retriever 和 Reranker 的關鍵，能提升模型辨別相似但不相關文本的能力。

```bash
# 先用基礎模型建立向量資料庫
python code/data_preparation/save_embeddings.py \
  --retriever_model_path intfloat/multilingual-e5-small \
  --build_db

# 挖掘 hard negatives
python code/data_preparation/mine_hard_negatives.py \
  --train_path data/train.txt \
  --index_path vector_database/passage_index.faiss \
  --sqlite_path vector_database/passage_store.db \
  --retriever_model intfloat/multilingual-e5-small \
  --topk 50 \
  --per_q_hard 2 \
  --out_path data/hardneg.jsonl
```

#### 檢查 Hard Negatives 品質
```bash
python code/check_tool/check_hardneg.py \
  --train_path data/train.txt \
  --hardneg_path data/hardneg.jsonl \
  --k 12 \
  --topk_check \
  --index_path vector_database/passage_index.faiss \
  --sqlite_path vector_database/passage_store.db \
  --retriever_model intfloat/multilingual-e5-small \
  --topk 50
```

---

### Step 3: 訓練 Retriever（雙塔模型）

Retriever 使用 Sentence Transformer 的雙塔架構，將 query 和 passage 編碼成向量。

```bash
python code/training/train_retriever.py \
  --train_path data/train.txt \
  --hardneg_path data/hardneg.jsonl \
  --corpus_path data/corpus.txt \
  --output_dir models/retriever \
  --base_model intfloat/multilingual-e5-small \
  --epochs 8 \
  --per_device_train_batch_size 192 \
  --gradient_accumulation_steps 2 \
  --max_seq_length 512 \
  --lr 2e-5 \
  --warmup_ratio 0.08 \
  --save_steps 50 \
  --eval_steps 50 \
  --logging_steps 10 \
  --seed 42
```

**關鍵參數說明：**
- `--epochs 8`: 訓練 8 個 epoch
- `--per_device_train_batch_size 192`: 每張 GPU 的 batch size（A100 建議 192）
- `--gradient_accumulation_steps 2`: 梯度累積步數（有效 batch size = 192 × 2 = 384）
- `--max_seq_length 512`: 最大序列長度
- `--lr 2e-5`: 學習率
- `--warmup_ratio 0.08`: warmup 比例（前 8% 的步數進行 warmup）

**訓練後視覺化：**
```bash
python code/training/plot_loss_curves_retriever.py \
  --model_dir models/retriever \
  --out_dir report_artifacts \
  --x_axis steps \
  --save_csv true
```

---

### Step 4: 訓練 Reranker（Cross-Encoder）

Reranker 使用 Cross-Encoder 架構，同時編碼 query 和 passage 的交互。

```bash
python code/training/train_rerank.py \
  --data_dir data \
  --output_dir models/reranker \
  --profile a100 \
  --epochs 3 \
  --lr 2e-5 \
  --batch_size 128 \
  --max_length 512 \
  --hard_neg_cap 4 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --logging_steps 100 \
  --num_workers 12 \
  --seed 12
```

**關鍵參數說明：**
- `--profile a100`: 硬體配置（可選 `a100`, `t4`, `auto`）
- `--batch_size 128`: 訓練 batch size（A100 建議 128）
- `--max_length 512`: Cross-Encoder 的最大輸入長度
- `--hard_neg_cap 4`: 每個 query 最多使用 4 個 hard negatives
- `--num_workers 12`: 資料載入的 worker 數量

**訓練後視覺化：**
```bash
python code/training/plot_loss_curves_rerank.py \
  --model_dir models/reranker \
  --out_dir report_artifacts \
  --x_axis steps \
  --save_csv true
```

---

### Step 5: 重建向量資料庫

使用訓練好的 Retriever 重新建立向量資料庫：

```bash
python code/data_preparation/save_embeddings.py \
  --data_folder ./data \
  --file_name corpus.txt \
  --output_folder ./vector_database \
  --retriever_model_path ./models/retriever \
  --output_index_file_name passage_index.faiss \
  --output_db_file_name passage_store.db \
  --batch_size 256 \
  --build_db
```

**參數說明：**
- `--build_db`: 同時建立 SQLite 資料庫（儲存原始文本）
- `--batch_size 256`: 編碼 batch size

---

## 📊 推論與評估

### 完整推論（Retriever + Reranker + Generator）

```bash
python code/evaluation/inference_batch.py \
  --data_folder ./data \
  --passage_file corpus.txt \
  --index_folder ./vector_database \
  --index_file passage_index.faiss \
  --sqlite_file passage_store.db \
  --test_data_path ./data/test_open.txt \
  --qrels_path ./data/qrels.txt \
  --retriever_model_path ./models/retriever \
  --reranker_model_path ./models/reranker \
  --generator_model Qwen/Qwen3-1.7B \
  --result_file_name result.json
```

**輸出格式 (`result.json`)：**
```json
{
  "records": [
    {
      "qid": "...",
      "query": "...",
      "generated": "...",
      "gold_answer": "...",
      "retrieved_passages": [...],
      "reranked_passages": [...]
    }
  ],
  "retrieval_metrics": {...},
  "generation_metrics": {...}
}
```

### 僅使用 Retriever（無 Reranker）

```bash
python code/evaluation/inference_batch_norerank.py \
  --retriever_model_path ./models/retriever \
  --test_data_path ./data/test_open.txt \
  --result_file_name result_norerank.json
```

---

## 🧰 工具說明

### 資料檢查工具 (`code/check_tool/`)

#### 1. `hw3_inspect_head.py` - JSONL 結構檢查器
快速查看 JSONL 檔案的 key 結構和值預覽。
```bash
python code/check_tool/hw3_inspect_head.py \
  --train data/train.txt \
  --corpus data/corpus.txt \
  --n 20
```

#### 2. `hw3_preview_qa.py` - 問答對預覽
專門查看 question、rewrite、answer 內容。
```bash
python code/check_tool/hw3_preview_qa.py \
  --train data/train.txt \
  --n 20
```

#### 3. `check_hardneg.py` - Hard Negatives 品質檢查
檢查挖掘的 hard negatives 是否與 gold answer 重複、是否在 Top-K 內等。
```bash
python code/check_tool/check_hardneg.py \
  --train_path data/train.txt \
  --hardneg_path data/hardneg.jsonl \
  --k 12 \
  --topk_check
```

#### 4. `check_train_corpus_overlap.py` - 資料重疊檢查
檢查訓練資料的 evidence 與 corpus 的重疊度。
```bash
python code/check_tool/check_train_corpus_overlap.py \
  --train data/train.txt \
  --corpus data/corpus.txt \
  --n 200
```

---

### 資料準備工具 (`code/data_preparation/`)

#### 1. `save_embeddings.py` - 建立向量資料庫
將 corpus 編碼成向量並建立 FAISS 索引。
```bash
python code/data_preparation/save_embeddings.py \
  --retriever_model_path intfloat/multilingual-e5-small \
  --build_db
```

#### 2. `mine_hard_negatives.py` - 挖掘 Hard Negatives
從 Top-K 檢索結果中挖掘 hard negatives。
```bash
python code/data_preparation/mine_hard_negatives.py \
  --train_path data/train.txt \
  --topk 50 \
  --per_q_hard 2 \
  --out_path data/hardneg.jsonl
```

---

### 評估工具 (`code/evaluation/`)

#### 1. `scan_train_stats.py` - 訓練資料統計
深度分析訓練資料的統計特性（token 長度、標籤分布等）。
```bash
python code/evaluation/scan_train_stats.py \
  --train_path data/train.txt \
  --limit 200 \
  --csv_out work/train_stats.csv
```

#### 2. `inference_batch.py` - 完整推論
完整的 RAG pipeline（Retriever + Reranker + Generator）。

#### 3. `inference_batch_norerank.py` - 簡化推論
僅使用 Retriever + Generator（無 Reranker）。

---

### 訓練工具 (`code/training/`)

#### 1. `train_retriever.py` - Retriever 訓練
訓練雙塔模型（Bi-Encoder）。

#### 2. `train_rerank.py` - Reranker 訓練
訓練 Cross-Encoder 重排模型。

#### 3. `plot_loss_curves_retriever.py` - Retriever 曲線視覺化
繪製 Retriever 的訓練/驗證 loss 曲線。
```bash
python code/training/plot_loss_curves_retriever.py \
  --model_dir models/retriever \
  --out_dir report_artifacts
```

#### 4. `plot_loss_curves_rerank.py` - Reranker 曲線視覺化
繪製 Reranker 的訓練/驗證 loss 曲線。
```bash
python code/training/plot_loss_curves_rerank.py \
  --model_dir models/reranker \
  --out_dir report_artifacts
```

---

## 🔬 實驗重現

### 完整實驗流程（從頭開始）

```bash
# ========== 1. 環境準備 ==========
pip install -r requirements.txt
echo 'hf_token="your_token"' > .env

# ========== 2. 資料檢查 ==========
# 檢查資料結構
python code/check_tool/hw3_inspect_head.py --train data/train.txt --n 20
python code/check_tool/hw3_preview_qa.py --train data/train.txt --n 20

# 統計分析
python code/evaluation/scan_train_stats.py \
  --train_path data/train.txt \
  --limit 200 \
  --csv_out work/train_stats.csv

# ========== 3. 挖掘 Hard Negatives ==========
# 先用基礎模型建立向量庫
python code/data_preparation/save_embeddings.py \
  --retriever_model_path intfloat/multilingual-e5-small \
  --build_db

# 挖掘 hard negatives
python code/data_preparation/mine_hard_negatives.py \
  --train_path data/train.txt \
  --index_path vector_database/passage_index.faiss \
  --sqlite_path vector_database/passage_store.db \
  --retriever_model intfloat/multilingual-e5-small \
  --topk 50 \
  --per_q_hard 2 \
  --out_path data/hardneg.jsonl

# 檢查 hard negatives 品質
python code/check_tool/check_hardneg.py \
  --train_path data/train.txt \
  --hardneg_path data/hardneg.jsonl \
  --k 12 \
  --topk_check

# ========== 4. 訓練 Retriever ==========
python code/training/train_retriever.py \
  --train_path data/train.txt \
  --hardneg_path data/hardneg.jsonl \
  --corpus_path data/corpus.txt \
  --output_dir models/retriever \
  --base_model intfloat/multilingual-e5-small \
  --epochs 8 \
  --per_device_train_batch_size 192 \
  --gradient_accumulation_steps 2 \
  --max_seq_length 512 \
  --lr 2e-5 \
  --warmup_ratio 0.08 \
  --save_steps 50 \
  --eval_steps 50 \
  --logging_steps 10 \
  --seed 42

# 視覺化訓練曲線
python code/training/plot_loss_curves_retriever.py \
  --model_dir models/retriever \
  --out_dir report_artifacts

# ========== 5. 訓練 Reranker ==========
python code/training/train_rerank.py \
  --data_dir data \
  --output_dir models/reranker \
  --profile a100 \
  --epochs 3 \
  --lr 2e-5 \
  --batch_size 128 \
  --max_length 512 \
  --hard_neg_cap 4 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --logging_steps 100 \
  --num_workers 12 \
  --seed 12

# 視覺化訓練曲線
python code/training/plot_loss_curves_rerank.py \
  --model_dir models/reranker \
  --out_dir report_artifacts

# ========== 6. 重建向量資料庫 ==========
python code/data_preparation/save_embeddings.py \
  --retriever_model_path ./models/retriever \
  --build_db

# ========== 7. 推論與評估 ==========
# 完整推論（含 Reranker）
python code/evaluation/inference_batch.py \
  --retriever_model_path ./models/retriever \
  --reranker_model_path ./models/reranker \
  --test_data_path ./data/test_open.txt \
  --result_file_name result.json

# 簡化推論（無 Reranker）- 用於消融實驗
python code/evaluation/inference_batch_norerank.py \
  --retriever_model_path ./models/retriever \
  --test_data_path ./data/test_open.txt \
  --result_file_name result_norerank.json
```

---

## 📁 目錄結構

```
ADL-HW3/
├── code/                          # 核心程式碼
│   ├── check_tool/                # 資料檢查工具
│   │   ├── check_hardneg.py
│   │   ├── check_train_corpus_overlap.py
│   │   ├── hw3_inspect_head.py
│   │   └── hw3_preview_qa.py
│   ├── data_preparation/          # 資料準備
│   │   ├── mine_hard_negatives.py
│   │   └── save_embeddings.py
│   ├── evaluation/                # 推論與評估
│   │   ├── inference_batch.py
│   │   ├── inference_batch_norerank.py
│   │   ├── scan_train_stats.py
│   │   └── utils.py
│   └── training/                  # 訓練腳本
│       ├── plot_loss_curves_rerank.py
│       ├── plot_loss_curves_retriever.py
│       ├── train_rerank.py
│       └── train_retriever.py
├── data/                          # 資料集
│   ├── corpus.txt                 # 段落語料庫
│   ├── qrels.txt                  # 相關性標註
│   ├── test_open.txt              # 測試資料
│   ├── train.txt                  # 訓練資料
│   └── hardneg.jsonl              # (生成) Hard negatives
├── vector_database/               # 向量資料庫
│   ├── passage_index.faiss        # FAISS 索引
│   └── passage_store.db           # SQLite 文本儲存
├── models/                        # 訓練模型
│   ├── retriever/                 # Retriever 模型
│   └── reranker/                  # Reranker 模型
├── report_artifacts/              # (生成) 報告產出
│   ├── retriever_loss_curve.png
│   └── reranker_loss_curve.png
├── download.sh                    # 模型下載腳本
├── requirements.txt               # Python 套件清單
├── .env                           # (需建立) HuggingFace token
└── README.md                      # 本文件
```

---

## 📝 注意事項

### 硬體需求
- **Retriever 訓練**: 建議使用 A100 (40GB/80GB) 或 2×RTX 4090
- **Reranker 訓練**: 建議使用 A100 或單張 RTX 3090/4090
- **推論**: 至少需要 16GB VRAM

### Batch Size 調整
根據您的 GPU 記憶體調整 batch size：

| GPU | Retriever Batch Size | Reranker Batch Size |
|-----|---------------------|---------------------|
| A100 (80GB) | 192-256 | 128-192 |
| A100 (40GB) | 128-192 | 96-128 |
| RTX 4090 (24GB) | 64-96 | 48-64 |
| RTX 3090 (24GB) | 64-96 | 48-64 |
| T4 (16GB) | 32-48 | 24-32 |

### 訓練時間估計
- **Retriever**: 約 4-6 小時 (A100, 8 epochs)
- **Reranker**: 約 2-3 小時 (A100, 3 epochs)
- **向量資料庫建立**: 約 10-15 分鐘
- **推論** (300 queries): 約 30-45 分鐘

---

## 🐛 常見問題

### Q1: CUDA out of memory
**解決方法：** 降低 batch size 或增加 gradient accumulation steps
```bash
# 降低 batch size
--per_device_train_batch_size 64

# 或增加 gradient accumulation
--gradient_accumulation_steps 4
```

### Q2: HuggingFace token 錯誤
**解決方法：** 確認 `.env` 檔案格式正確
```bash
hf_token="hf_xxxxxxxxxxxxxxxxxxxxx"
```

### Q3: FAISS 索引載入失敗
**解決方法：** 重新建立向量資料庫
```bash
python code/data_preparation/save_embeddings.py \
  --retriever_model_path ./models/retriever \
  --build_db
```

### Q4: 推論速度太慢
**解決方法：** 調整 batch size 參數（在腳本內修改 `BATCH_Q` 和 `BATCH_GEN`）

---

## 📚 參考資料

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [E5 Model Paper](https://arxiv.org/abs/2212.03533)
- [Cross-Encoder for Re-ranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)

---

## 📧 聯絡資訊

如有問題，請聯絡：35049957a@gmail.com

---

## 📄 授權

本專案僅供學術用途，請勿用於商業用途。

---

**最後更新：** 2025-11-06
