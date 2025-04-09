#!/usr/bin/env python
"""
Script to run the pre-computation pipeline for IPL Fantasy Predictor.

This script processes all data and pre-computes statistics for:
1. Player performance (batting, bowling, recent form, venue performance)
2. Team head-to-head statistics
3. Venue statistics

Usage:
    python precompute_data.py [--data-dir DATA_DIR]

Options:
    --data-dir DATA_DIR    Path to the data directory (default: from settings)
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.precompute_pipeline import run_precompute_pipeline

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-compute data for IPL Fantasy Predictor"
    )
    parser.add_argument(
        "--data-dir", type=Path, help="Path to the data directory", default=None
    )

    args = parser.parse_args()

    # Run the pre-computation pipeline
    run_precompute_pipeline(args.data_dir)
