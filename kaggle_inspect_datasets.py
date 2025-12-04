"""
Simple Kaggle script to load IFViT datasets and inspect basic statistics.

Usage on Kaggle (in a notebook cell):

    !pip install -r /kaggle/working/IFViT/requirements.txt
    %cd /kaggle/working/IFViT
    !python kaggle_inspect_datasets.py

Assumes:
    - This repository is available under /kaggle/working/IFViT
    - Fingerprint datasets are mounted under /kaggle/input with paths:
        /kaggle/input/fvc2002/FVC2002
        /kaggle/input/fvc2004/FVC2004
        /kaggle/input/nist-sd300/NIST SD300
        /kaggle/input/nist-sd4/NIST4
        /kaggle/input/sd301a/images
        /kaggle/input/sd302a/images
"""

import os
import sys
from collections import Counter

import matplotlib.pyplot as plt

# Ensure local package is importable when running from /kaggle/working/IFViT
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data import (  # type: ignore
    PaperDatasetRoots,
    build_paper_train_entries,
    build_paper_val_entries,
    build_paper_test_entries,
    load_image,
    normalize_image,
)


def summarize_entries(name, entries):
    print(f"\n=== {name} ===")
    print(f"Total entries: {len(entries)}")
    if not entries:
        return

    # Count by dataset_name
    by_dataset = Counter(e.dataset_name for e in entries)
    print("By dataset_name:")
    for k, v in sorted(by_dataset.items()):
        print(f"  {k:25s}: {v}")

    # Count unique fingers
    unique_fingers = len({e.finger_global_id for e in entries})
    print(f"Unique global fingers: {unique_fingers}")


def show_samples(entries, n=6, title="Samples"):
    if not entries:
        print(f"No entries to display for {title}")
        return

    n = min(n, len(entries))
    plt.figure(figsize=(3 * n, 4))
    for i in range(n):
        e = entries[i]
        img = load_image(e.path)
        img = normalize_image(img)
        ax = plt.subplot(1, n, i + 1)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(f"{os.path.basename(e.path)}\n{e.dataset_name}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    print("Using dataset roots for Kaggle...")
    roots = PaperDatasetRoots()  # defaults point to /kaggle/input paths
    print(roots)

    print("\nBuilding train/val/test entries...")
    train_entries = build_paper_train_entries(roots)
    val_entries = build_paper_val_entries(roots)
    test_entries = build_paper_test_entries(roots)

    summarize_entries("TRAIN", train_entries)
    summarize_entries("VAL", val_entries)
    summarize_entries("TEST", test_entries)

    print("\nShowing a few sample images from TRAIN and TEST...")
    show_samples(train_entries, n=4, title="TRAIN samples")
    show_samples(test_entries, n=4, title="TEST samples")


if __name__ == "__main__":
    main()


