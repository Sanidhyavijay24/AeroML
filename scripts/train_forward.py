# -*- coding: utf-8 -*-
"""
@file train_forward.py
@description CLI script to run the forward ensemble model training
@module scripts
"""

import sys
import argparse
from pathlib import Path

# Bootstrap local src package imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aeroml.train import train_forward_ensemble


def main():
    parser = argparse.ArgumentParser(description="AeroML Forward Model Ensemble Training CLI")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,52,62",
        help="Comma-separated list of random seeds for the ensemble seeds (default: 42,52,62)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="cd_loss_only",
        choices=["cd_loss_only", "low_drag_only", "low_drag_plus_mach05"],
        help="Training variant defining loss/sample weighting strategy (default: cd_loss_only)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of epochs to train each seed model (default: 80)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for model training (default: 1024)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Forward_outputs/",
        help="Output directory to save trained model files and metrics (default: Forward_outputs/)"
    )

    args = parser.parse_args()

    # Parse seeds list
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("--- AeroML Forward Ensemble Training Configuration ---")
    print(f"Seeds:         {seeds}")
    print(f"Variant:       {args.variant}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Output Dir:    {args.output_dir}")
    print("-----------------------------------------------------")

    train_forward_ensemble(
        seeds=seeds,
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
