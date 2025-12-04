"""
Convenience script to run the full IFViT training pipeline on Kaggle (or locally)
with a single command.

It sequentially trains:
- Module 1: DenseRegModel (dense registration, L_D only)
- Module 2: MatcherModel  (global/local branches, L_D + L_E + L_A)

Usage example (inside Kaggle notebook or terminal):

    !python ifvit-jax/train_all.py \\
        --dataset_root /kaggle/input/YOUR_DATASET \\
        --num_classes 300

Make sure you have implemented the dataset loaders in `data.py`
(`dense_reg_dataset`, `matcher_dataset`, `preprocess_batch`, etc.)
and attached your fingerprint dataset in Kaggle under `/kaggle/input/...`.
"""

import argparse
from pathlib import Path

from config import DENSE_CONFIG, MATCH_CONFIG
from train_dense import train_dense_reg
from train_match import train_matcher


def main():
    parser = argparse.ArgumentParser(description="Run full IFViT training pipeline")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./data",
        help="Path to dataset root directory (e.g., /kaggle/input/your-dataset)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        help="Number of subject classes in the dataset (for ArcFace in Module 2)",
    )
    parser.add_argument(
        "--dense_checkpoint_dir",
        type=str,
        default=None,
        help="Optional override for Module 1 checkpoint directory",
    )
    parser.add_argument(
        "--matcher_checkpoint_dir",
        type=str,
        default=None,
        help="Optional override for Module 2 checkpoint directory",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1) Train Module 1: DenseRegModel
    # -------------------------------------------------------------------------
    dense_config = DENSE_CONFIG.copy()
    if args.dense_checkpoint_dir:
        dense_config["checkpoint_dir"] = args.dense_checkpoint_dir

    print("\n" + "=" * 80)
    print("TRAINING MODULE 1: Dense Registration (DenseRegModel)")
    print("=" * 80 + "\n")

    train_dense_reg(
        dataset_root=args.dataset_root,
        config=dense_config,
        resume_from=None,
    )

    # Use final checkpoint from Module 1 as input to Module 2
    dense_ckpt_dir = Path(dense_config["checkpoint_dir"])
    dense_final_ckpt = str(dense_ckpt_dir / "dense_reg_ckpt.pkl")

    # -------------------------------------------------------------------------
    # 2) Train Module 2: MatcherModel
    # -------------------------------------------------------------------------
    match_config = MATCH_CONFIG.copy()
    if args.matcher_checkpoint_dir:
        match_config["checkpoint_dir"] = args.matcher_checkpoint_dir

    # Tell Module 2 where to load the trained transformer from
    match_config["dense_reg_ckpt"] = dense_final_ckpt

    print("\n" + "=" * 80)
    print("TRAINING MODULE 2: Matcher (global + local branches)")
    print("=" * 80 + "\n")

    train_matcher(
        dataset_root=args.dataset_root,
        num_classes=args.num_classes,
        config=match_config,
        resume_from=None,
    )

    print("\n" + "=" * 80)
    print("FULL IFViT TRAINING PIPELINE COMPLETED")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


