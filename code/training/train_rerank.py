from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
# import torch.nn.functional as F   # ← 目前不自訂 metrics，就先不需要
from datasets import Dataset, DatasetDict, Features, Value, load_from_disk
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from transformers import TrainerCallback  # EvalPrediction 不用就先拿掉

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
BR_TAG_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a CrossEncoder reranker on ADL-style data.")
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--cache_dir", default="work/pairs", type=str)
    parser.add_argument("--output_dir", default="models/reranker", type=str)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--batch_size", default=-1, type=int)
    parser.add_argument("--profile", default="auto", choices=["auto", "a100", "t4"])
    parser.add_argument("--max_length", default=512, type=int)  # 固定 512
    parser.add_argument("--eval_steps", default=1000, type=int)
    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--logging_steps", default=100, type=int)
    parser.add_argument("--num_workers", default=-1, type=int)
    parser.add_argument("--seed", default=12, type=int)
    parser.add_argument("--hard_neg_cap", default=4, type=int, help="每 qid 最多加入多少條 hard negatives")
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = BR_TAG_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def preprocess_text(text: str) -> str | None:
    if text is None:
        return None
    if len(text) > 4000:
        return None
    cleaned = clean_text(text)
    if len(cleaned) < 3:
        return None
    return cleaned


def load_hardneg_map(path: Path) -> dict:
    """讀取由 mine_hard_negatives 產生的 data/hardneg.jsonl。"""
    if not path.exists():
        return {}
    m = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            m[str(obj["qid"])] = obj.get("hard_negatives") or []
    return m


def build_pairs_from_train(
    train_path: Path,
    seed: int,
    hard_map: Dict[str, List[str]],
    hard_neg_cap: int = 4,
) -> Dict[str, List[Dict[str, float | str]]]:
    """
    僅使用 train.txt：
    - retrieval_labels==1 的 evidence → 正例（每行 1）
    - retrieval_labels==0 的 evidence → 易負（每行 4）
    - 額外再加 hard_map[qid] 中的前 hard_neg_cap 條（去重後）
    """
    def norm_label(v) -> int | None:
        try:
            return int(v)
        except Exception:
            try:
                return int(round(float(v)))
            except Exception:
                return None

    rng = random.Random(seed)
    pairs_by_qid: Dict[str, List[Dict[str, float | str]]] = {}

    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            qid = str(rec.get("qid", ""))
            if not qid:
                continue

            query_text = preprocess_text(rec.get("rewrite") or rec.get("question") or "")
            if query_text is None:
                continue

            evidences = rec.get("evidences") or []
            labels = rec.get("retrieval_labels") or []
            if not evidences or not labels or len(evidences) != len(labels):
                continue

            pos_text = None
            neg_list: List[str] = []

            for ev, lb in zip(evidences, labels):
                if not isinstance(ev, str):
                    continue
                ev_clean = preprocess_text(ev)
                if ev_clean is None:
                    continue
                lab = norm_label(lb)
                if lab == 1:
                    pos_text = ev_clean
                elif lab == 0:
                    neg_list.append(ev_clean)

            if not pos_text:
                # 你的統計顯示不會發生；安全起見跳過
                continue

            q_pairs = pairs_by_qid.setdefault(qid, [])
            used = set()

            # 1 正
            q_pairs.append({"query": query_text, "passage": pos_text, "label": 1.0})
            used.add(pos_text)

            # 4 易負
            for neg in neg_list:
                if neg and neg not in used:
                    q_pairs.append({"query": query_text, "passage": neg, "label": 0.0})
                    used.add(neg)

            # + hard_neg_cap 難負（依序加入，避免重複）
            hard_list = hard_map.get(qid, [])
            added_hard = 0
            for h in hard_list:
                if not h:
                    continue
                h_clean = preprocess_text(h)
                if not h_clean or h_clean in used:
                    continue
                q_pairs.append({"query": query_text, "passage": h_clean, "label": 0.0})
                used.add(h_clean)
                added_hard += 1
                if added_hard >= max(0, hard_neg_cap):
                    break

    return pairs_by_qid


def split_pairs(
    pairs_by_qid: Dict[str, List[Dict[str, float | str]]],
    seed: int,
) -> Tuple[List[Dict[str, float | str]], List[Dict[str, float | str]]]:
    if not pairs_by_qid:
        raise ValueError("No training pairs were generated from the provided data.")

    rng = random.Random(seed)
    qids = list(pairs_by_qid.keys())

    if len(qids) > 1:
        rng.shuffle(qids)
        split_idx = max(1, int(len(qids) * 0.9))
        if split_idx >= len(qids):
            split_idx = len(qids) - 1
        train_qids = set(qids[:split_idx])
        valid_qids = set(qids[split_idx:])
        train_pairs = [p for q in train_qids for p in pairs_by_qid[q]]
        valid_pairs = [p for q in valid_qids for p in pairs_by_qid[q]]

        if not any(p["label"] == 0.0 for p in valid_pairs):
            for q in list(train_qids):
                cand = pairs_by_qid[q]
                if any(p["label"] == 0.0 for p in cand):
                    train_qids.remove(q)
                    valid_qids.add(q)
                    train_pairs = [pp for qq in train_qids for pp in pairs_by_qid[qq]]
                    valid_pairs = [pp for qq in valid_qids for pp in pairs_by_qid[qq]]
                    if any(p["label"] == 0.0 for p in valid_pairs):
                        break

        if not valid_pairs:
            raise ValueError("Validation split ended up empty. Check the input data.")
    else:
        sole_qid = qids[0]
        sole_pairs = pairs_by_qid[sole_qid][:]
        rng.shuffle(sole_pairs)
        val_size = max(1, int(len(sole_pairs) * 0.1))
        if val_size >= len(sole_pairs):
            val_size = max(1, len(sole_pairs) // 2)
        valid_pairs = sole_pairs[:val_size]
        train_pairs = sole_pairs[val_size:]
        if not train_pairs:
            raise ValueError("Not enough examples to build a train split from a single query.")

    if not any(p["label"] == 1.0 for p in valid_pairs):
        raise ValueError("Validation split must contain positive examples.")
    if not any(p["label"] == 0.0 for p in valid_pairs):
        raise ValueError("Validation split must contain negative examples.")

    return train_pairs, valid_pairs


def save_dataset_to_cache(
    train_pairs: List[Dict[str, float | str]],
    valid_pairs: List[Dict[str, float | str]],
    cache_dir: Path,
    seed: int,
) -> Tuple[Dataset, Dataset]:
    features = Features({"query": Value("string"), "passage": Value("string"), "label": Value("float32")})
    train_dataset = Dataset.from_list(train_pairs, features=features).shuffle(seed=seed)
    valid_dataset = Dataset.from_list(valid_pairs, features=features)

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    dataset_dict = DatasetDict({"train": train_dataset, "valid": valid_dataset})
    dataset_dict.save_to_disk(str(cache_dir))
    return train_dataset, valid_dataset


def load_adl_data(data_dir: str, cache_dir: str, seed: int, hard_neg_cap: int) -> Tuple[Dataset, Dataset]:
    cache_path = Path(cache_dir)
    if cache_path.exists():
        try:
            dataset_dict = load_from_disk(str(cache_path))
            LOGGER.info("Loaded cached dataset from %s", cache_path)
            return dataset_dict["train"], dataset_dict["valid"]
        except Exception as exc:
            LOGGER.warning("Failed to load cached dataset at %s (%s). Rebuilding cache.", cache_path, exc)
            shutil.rmtree(cache_path)

    data_path = Path(data_dir)
    hard_map = load_hardneg_map(data_path / "hardneg.jsonl")
    pairs_by_qid = build_pairs_from_train(data_path / "train.txt", seed, hard_map, hard_neg_cap=hard_neg_cap)
    train_pairs, valid_pairs = split_pairs(pairs_by_qid, seed)

    all_pairs = train_pairs + valid_pairs
    pos_cnt = sum(1 for p in all_pairs if p["label"] == 1.0)
    neg_cnt = sum(1 for p in all_pairs if p["label"] == 0.0)
    total = pos_cnt + neg_cnt
    pos_ratio = pos_cnt / max(1, total)
    LOGGER.info("Pairs summary: total=%d | positives=%d | negatives=%d | pos_ratio=%.3f", total, pos_cnt, neg_cnt, pos_ratio)
    LOGGER.info("Prepared %d training pairs and %d validation pairs.", len(train_pairs), len(valid_pairs))
    return save_dataset_to_cache(train_pairs, valid_pairs, cache_path, seed)


def detect_profile() -> str:
    if not torch.cuda.is_available():
        return "t4"
    name = torch.cuda.get_device_name(0).upper()
    if "A100" in name:
        return "a100"
    if "T4" in name:
        return "t4"
    return "t4"


def resolve_training_profile(args: argparse.Namespace) -> Tuple[str, Dict[str, int | float | bool]]:
    profile = args.profile
    if profile == "auto":
        profile = detect_profile()

    defaults = {
        "a100": {
            "train_batch_size": 96,   # 放大
            "eval_batch_size": 128,
            "gradient_accumulation_steps": 1,
            "bf16": True,
            "fp16": False,
            "num_workers": 12,
        },
        "t4": {
            "train_batch_size": 16,
            "eval_batch_size": 16,
            "gradient_accumulation_steps": 2,
            "bf16": False,
            "fp16": True,
            "num_workers": 4,
        },
    }
    config = defaults[profile].copy()

    if args.batch_size and args.batch_size > 0:
        config["train_batch_size"] = args.batch_size
        config["eval_batch_size"] = args.batch_size
    if args.num_workers != -1:
        config["num_workers"] = args.num_workers
    if not torch.cuda.is_available():
        config["bf16"] = False
        config["fp16"] = False
    return profile, config


class CSVLoggingCallback(TrainerCallback):
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            self.csv_path.write_text("step,epoch,train_loss,valid_loss,valid_acc,lr,time\n", encoding="utf-8")
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if logs is None:
            return
        elapsed = time.time() - self.start_time
        step = state.global_step if state.global_step is not None else ""
        epoch = state.epoch if state.epoch is not None else ""
        train_loss = logs.get("loss", "")
        valid_loss = logs.get("eval_loss", "")  # 用預設 eval_loss
        valid_acc = logs.get("eval_acc", "")    # 若無則為空
        lr = logs.get("learning_rate", "")
        row = [
            _format_csv_value(step),
            _format_csv_value(epoch),
            _format_csv_value(train_loss),
            _format_csv_value(valid_loss),
            _format_csv_value(valid_acc),
            _format_csv_value(lr),
            f"{elapsed:.2f}",
        ]
        with self.csv_path.open("a", encoding="utf-8") as csv_file:
            csv_file.write(",".join(row) + "\n")


def _format_csv_value(value) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_dataset, valid_dataset = load_adl_data(
        args.data_dir,
        args.cache_dir,
        args.seed,
        args.hard_neg_cap,
    )

    profile_name, profile_config = resolve_training_profile(args)
    LOGGER.info(
        "Using training profile '%s' with batch size %s, gradient_accumulation_steps=%s, fp16=%s, bf16=%s, num_workers=%s",
        profile_name,
        profile_config["train_batch_size"],
        profile_config["gradient_accumulation_steps"],
        profile_config["fp16"],
        profile_config["bf16"],
        profile_config["num_workers"],
    )

    # 固定 512，並只截斷 passage（second）
    model = CrossEncoder(
        MODEL_NAME,
        num_labels=1,
        max_length=args.max_length,
        tokenizer_kwargs={"truncation": "only_second"},
    )
    model.max_length = args.max_length
    loss = BinaryCrossEntropyLoss(model)

    training_args = CrossEncoderTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=profile_config["train_batch_size"],
        per_device_eval_batch_size=profile_config["eval_batch_size"],
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=profile_config["fp16"],
        bf16=profile_config["bf16"],
        gradient_accumulation_steps=profile_config["gradient_accumulation_steps"],
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",     # 保守用內建 eval_loss (BCE)
        greater_is_better=False,
        seed=args.seed,
        dataloader_num_workers=profile_config["num_workers"],
        dataloader_pin_memory=True,            # 新增：更穩
        eval_accumulation_steps=8,             # 新增：避免 eval 尖峰
        report_to=[],                          # 關閉外部後端
    )

    csv_logger = CSVLoggingCallback(Path("work/train_logs.csv"))

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        loss=loss,
    )
    trainer.add_callback(csv_logger)

    trainer.train()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_path)
    LOGGER.info("Model saved to %s", output_path)


if __name__ == "__main__":
    main()
