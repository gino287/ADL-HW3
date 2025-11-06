#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, random, sys, re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset
from sentence_transformers import (
    InputExample, SentenceTransformer, SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments, losses, models,
)

try:
    import orjson
except Exception:
    orjson = None  # type: ignore


# ------------------------- Utils -------------------------
def jloads(s: str) -> Any:
    if orjson is not None:
        return orjson.loads(s)
    return json.loads(s)

def set_global_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def label_is_positive(v: Any) -> bool:
    if isinstance(v, bool): return v
    try: return float(v) > 0
    except Exception: return False


# ------------------------- Data loading (train only) -------------------------
def extract_query(rec: Mapping[str, Any]) -> str | None:
    q = (rec.get("rewrite") or rec.get("question") or "").strip()
    return q or None

def resolve_evidence_text(e: Any) -> str:
    # 支援字串或物件 {"text":..., "id":...}
    if isinstance(e, str):
        return clean_ws(e)
    if isinstance(e, Mapping):
        txt = e.get("text") or ""
        if txt:
            return clean_ws(str(txt))
        # 沒文字就回空（訓練不用 id）
    return ""

def gather_positive_passages(rec: Mapping[str, Any]) -> List[str]:
    positives: List[str] = []
    evs = rec.get("evidences") or []
    labs = rec.get("retrieval_labels") or []
    if isinstance(labs, list):
        for ev, lab in zip(evs, labs):
            if label_is_positive(lab):
                t = resolve_evidence_text(ev)
                if t:
                    positives.append(t)
    elif isinstance(labs, MutableMapping):
        # 少見：若 labels 是 dict，通常 evs 也會是對應 id；此資料集幾乎用不到
        for _doc_id, rel in labs.items():
            if label_is_positive(rel):
                # 訓練不使用 id 對齊，僅保留文字
                pass
    # 去重＆去空＆過短
    uniq = []
    seen = set()
    for p in positives:
        p = clean_ws(p)
        if len(p) < 10:  # 太短的段落多半是噪音
            continue
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq

def build_training_rows(train_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    total_q = 0
    with train_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            rec = jloads(line)
            query = extract_query(rec)
            if not query: continue
            positives = gather_positive_passages(rec)
            if not positives: continue
            total_q += 1
            anchor = f"query: {clean_ws(query)}"
            for pos in positives:
                rows.append({"anchor": anchor, "positive": f"passage: {pos}"})
    if not rows:
        raise RuntimeError(f"No positive pairs from {train_path}")
    print(f"[INFO] Loaded {total_q} queries with positives -> {len(rows)} pairs.")
    return rows

def prepare_datasets(rows: Sequence[Dict[str, str]]) -> Tuple[Dataset, Dataset]:
    ds = Dataset.from_list(list(rows))
    # 10% 驗證；若資料很小就退到1筆
    try:
        sp = ds.train_test_split(test_size=0.1, seed=42, shuffle=True)
    except ValueError:
        sp = ds.train_test_split(test_size=1, seed=42, shuffle=True)
    return sp["train"], sp["test"]


# ------------------------- Collator -------------------------
class SmartBatchingCollator:
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.valid_label_columns = ["labels", "label"]

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, Any]:
        exs = [InputExample(texts=[it["anchor"], it["positive"]]) for it in features]
        sent_feats, labels = self.model.smart_batching_collate(exs)
        batch: Dict[str, Any] = {"return_loss": True}
        for i, feat in enumerate(sent_feats):
            for k, v in feat.items():
                batch[f"sentence_{i}_{k}"] = v
        if labels is not None:
            batch["labels"] = labels  # ！！關鍵：用複數
        return batch


# ------------------------- Model / Training helpers -------------------------
def build_model(model_name: str, max_seq_len: int) -> SentenceTransformer:
    transformer = models.Transformer(model_name, max_seq_length=max_seq_len)
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True, pooling_mode_cls_token=False, pooling_mode_max_tokens=False,
    )
    normalize = models.Normalize()
    return SentenceTransformer(modules=[transformer, pooling, normalize])

def collect_loss_history(log_history: Iterable[Dict[str, Any]]) -> Tuple[List[Tuple[float,float]], List[Tuple[float,float]]]:
    tr, ev = [], []
    for e in log_history:
        step = float(e.get("step", e.get("epoch", 0.0)))
        if "loss" in e: tr.append((step, float(e["loss"])))
        if "eval_loss" in e: ev.append((step, float(e["eval_loss"])))
    return tr, ev

def plot_losses(tr, ev, out: Path) -> None:
    if not tr and not ev: return
    plt.figure(figsize=(8,5))
    if tr:
        x,y = zip(*tr); plt.plot(x,y,label="train_loss")
    if ev:
        x,y = zip(*ev); plt.plot(x,y,label="eval_loss")
    plt.xlabel("Step/Epoch"); plt.ylabel("Loss"); plt.title("Training & Eval Loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True); plt.savefig(out); plt.close()

def pick_precision() -> Tuple[bool, bool]:
    try:
        import torch
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return True, False    # A100/L4/L40 → bf16
            return False, True        # 其他卡 → fp16
    except Exception:
        pass
    return False, False


# ------------------------- Main -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train a SentenceTransformer retriever (train.txt only).")
    p.add_argument("--data_dir", type=Path, default=Path("./data"))
    p.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-small")
    p.add_argument("--output_dir", type=Path, default=Path("./models/retriever"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=64)  # 穩跑預設
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_seq_length", type=int, default=384)
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--resume_from", type=Path, default=None)

    return p.parse_args()

def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_global_seed(args.seed)

    train_path = args.data_dir / "train.txt"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")

    print("[INFO] Building training pairs from train.txt ...")
    rows = build_training_rows(train_path)
    train_ds, eval_ds = prepare_datasets(rows)
    print(f"[INFO] Train pairs: {len(train_ds)} | Eval pairs: {len(eval_ds)}")

    model = build_model(args.model_name, args.max_seq_length)
    model.max_seq_length = args.max_seq_length
    loss = losses.MultipleNegativesRankingLoss(model)

    use_bf16, use_fp16 = pick_precision()

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        do_train=True, do_eval=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=use_bf16, fp16=use_fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        eval_strategy="steps", eval_steps=args.eval_steps,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_strategy="steps", logging_steps=args.logging_steps,
        remove_unused_columns=False,
        seed=args.seed,
        report_to=["tensorboard"],
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        loss=loss, data_collator=SmartBatchingCollator(model),
    )

    print("[INFO] Starting training ...")
    resume = str(args.resume_from) if args.resume_from else None
    train_result = trainer.train(resume_from_checkpoint=resume)

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save(str(args.output_dir))

    print("[INFO] Evaluating best checkpoint ...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    tr_pts, ev_pts = collect_loss_history(trainer.state.log_history)
    plot_losses(tr_pts, ev_pts, Path(args.output_dir) / "loss_curve.png")

    best_ckpt = trainer.state.best_model_checkpoint or str(args.output_dir)
    fin = eval_metrics.get("eval_loss")
    if fin is not None:
        print(f"[INFO] Done. Best checkpoint: {best_ckpt}. Final eval_loss: {fin:.4f}")
    else:
        print(f"[INFO] Done. Best checkpoint: {best_ckpt}.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr); sys.exit(1)
