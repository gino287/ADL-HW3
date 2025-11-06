#!/usr/bin/env python3
"""
Plot training/eval loss curves from SentenceTransformer trainer_state.json.

Usage:
  python plot_loss_curves.py --model_dir ./models/retriever \
                             --out_dir ./report_artifacts \
                             --x_axis steps \
                             --smooth 0 \
                             --save_csv true
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"trainer_state.json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def collect_points(log_history: List[Dict[str, Any]], x_axis: str) -> Tuple[List[Tuple[float,float]], List[Tuple[float,float]]]:
    """
    x_axis in {"steps","epochs"}; ST v5 通常有 'step' 與 'epoch'
    """
    train_pts, eval_pts = [], []
    for e in log_history:
        x = None
        if x_axis == "steps" and "step" in e:
            x = float(e["step"])
        elif x_axis == "epochs" and "epoch" in e:
            x = float(e["epoch"])
        if x is None:
            continue
        if "loss" in e:
            train_pts.append((x, float(e["loss"])))
        if "eval_loss" in e:
            eval_pts.append((x, float(e["eval_loss"])))
    # 依 x 排序，避免線條跳動
    train_pts.sort(key=lambda t: t[0])
    eval_pts.sort(key=lambda t: t[0])
    return train_pts, eval_pts

def moving_average(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(y) == 0:
        return y
    k = min(k, len(y))
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(y, kernel, mode="same")

def plot_curves(train_pts, eval_pts, out_png: Path, x_axis: str, smooth: int) -> None:
    plt.figure(figsize=(8, 5))
    if train_pts:
        x, y = np.array([p[0] for p in train_pts]), np.array([p[1] for p in train_pts], dtype=float)
        if smooth and len(y) > 3:
            y = moving_average(y, smooth)
        plt.plot(x, y, label="train_loss")
    if eval_pts:
        x, y = np.array([p[0] for p in eval_pts]), np.array([p[1] for p in eval_pts], dtype=float)
        if smooth and len(y) > 3:
            y = moving_average(y, max(1, smooth // 2))
        plt.plot(x, y, label="eval_loss")
    plt.xlabel("Steps" if x_axis=="steps" else "Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Evaluation Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def save_csv(train_pts, eval_pts, out_csv: Path) -> None:
    lines = ["type,x,y"]
    lines += [f"train,{x:.6f},{y:.6f}" for x,y in train_pts]
    lines += [f"eval,{x:.6f},{y:.6f}" for x,y in eval_pts]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(lines), encoding="utf-8")

def parse_args():
    ap = argparse.ArgumentParser(description="Plot loss curves from trainer_state.json")
    ap.add_argument("--model_dir", type=Path, default=Path("./models/retriever"))
    ap.add_argument("--out_dir", type=Path, default=Path("./report_artifacts"))
    ap.add_argument("--x_axis", choices=["steps","epochs"], default="steps",
                    help="x 軸選 steps 或 epochs（建議 steps）")
    ap.add_argument("--smooth", type=int, default=0, help="移動平均視窗，0=不平滑；建議 5~11")
    ap.add_argument("--save_csv", type=lambda x: str(x).lower()!="false", default=True)
    return ap.parse_args()

def main():
    args = parse_args()
    state_path = args.model_dir / "trainer_state.json"
    state = load_state(state_path)
    log_history = state.get("log_history", [])
    if not isinstance(log_history, list) or not log_history:
        raise RuntimeError(f"log_history is empty in {state_path}")

    train_pts, eval_pts = collect_points(log_history, x_axis=args.x_axis)
    if not train_pts and not eval_pts:
        raise RuntimeError("No train/eval points found. 檢查 logging_steps / eval_steps 是否過大。")

    out_png = args.out_dir / f"loss_curve_{args.x_axis}.png"
    plot_curves(train_pts, eval_pts, out_png, args.x_axis, args.smooth)

    if args.save_csv:
        save_csv(train_pts, eval_pts, args.out_dir / f"loss_points_{args.x_axis}.csv")

    print(f"[DONE] Saved plot: {out_png}")
    if args.save_csv:
        print(f"[DONE] Saved CSV:  {args.out_dir / ('loss_points_' + args.x_axis + '.csv')}")

if __name__ == "__main__":
    main()
