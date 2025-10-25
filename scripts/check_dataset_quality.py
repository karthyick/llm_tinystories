#!/usr/bin/env python3
"""Standalone script to check dataset quality.

Usage:
    python scripts/check_dataset_quality.py --dataset tinystories
    python scripts/check_dataset_quality.py --dataset wikitext-103 --strict
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.quality_checker import check_dataset_quality
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)


def main():
    parser = argparse.ArgumentParser(description="Check dataset quality before training")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["tinystories", "wikitext-103"],
        help="Dataset to check"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to check"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of samples to check (default: 10000, use 0 for all)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any quality issues (default: allow warnings)"
    )

    args = parser.parse_args()

    sample_size = args.sample_size if args.sample_size > 0 else None

    print("\n" + "=" * 70)
    print("DATASET QUALITY CHECK")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Sample Size: {sample_size or 'ALL'}")
    print(f"Strict Mode: {args.strict}")
    print("=" * 70 + "\n")

    passed = check_dataset_quality(
        dataset_name=args.dataset,
        split=args.split,
        sample_size=sample_size,
        strict=args.strict,
    )

    print("\n" + "=" * 70)
    if passed:
        print("✅ QUALITY CHECK PASSED")
        print("Dataset is ready for training!")
    else:
        print("❌ QUALITY CHECK FAILED")
        print("Please review the issues above before training.")
    print("=" * 70 + "\n")

    return 0 if passed else 1


if __name__ == "__main__":
    exit(main())
