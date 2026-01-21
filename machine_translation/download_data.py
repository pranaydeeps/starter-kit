#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Downloader for Machine Translation

This script allows you to download and save datasets from the Hugging Face Hub
in the format used for translation experiments.

Usage:
    python download_data.py --repo_name LT3/nfr_bt_nmt_english-french --base_path data/en-fr
"""

import argparse
from datasets import load_dataset
import os


def save_data(data, file_path):
    """Save a list of strings to a text file, one per line."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(data) + "\n")


def download_and_save_dataset(repo_name, base_path):
    """
    Download a dataset from Hugging Face Hub and save it as text files.
    
    Args:
        repo_name: Repository name on Hugging Face (e.g., 'LT3/nfr_bt_nmt_english-french')
        base_path: Base path where the dataset files will be saved
    """
    # Load the dataset from Hugging Face Hub
    print(f"Loading dataset from: {repo_name}")
    dataset = load_dataset(repo_name)

    # Ensure the necessary directory exists
    os.makedirs(base_path, exist_ok=True)

    # Dictionary to store dataset paths
    dataset_paths = {}

    # Save the datasets to disk
    for split in dataset.keys():
        # Handle mono splits specially
        if "mono_english" in split or "mono_ukrainian" in split or "mono_french" in split:
            lang_code = "en" if "english" in split else ("uk" if "ukrainian" in split else "fr")
            feature = "english" if "english" in split else ("ukrainian" if "ukrainian" in split else "french")
            if feature in dataset[split].column_names:
                path = f"{base_path}/{lang_code}_mono.txt"
                save_data(dataset[split][feature], path)
                dataset_paths[f"{lang_code}_mono"] = path
        else:
            # Save data for other splits
            for feature in ["english", "french", "ukrainian"]:
                if feature in dataset[split].column_names:
                    lang_code = "en" if feature == "english" else ("fr" if feature == "french" else "uk")
                    path = f"{base_path}/{lang_code}_{split}.txt"
                    save_data(dataset[split][feature], path)
                    dataset_paths[f"{lang_code}_{split}"] = path

    print("Dataset saved to:")
    for key, path in dataset_paths.items():
        print(f"  {key}: {path}")
    
    return dataset_paths


def main():
    parser = argparse.ArgumentParser(
        description="Download and save datasets from Hugging Face."
    )
    parser.add_argument(
        "--repo_name",
        required=True,
        help="Repository name on Hugging Face (e.g., 'LT3/nfr_bt_nmt_english-french')",
    )
    parser.add_argument(
        "--base_path",
        required=True,
        help="Base path where the dataset files will be saved (e.g., 'data/en-fr')",
    )
    args = parser.parse_args()

    download_and_save_dataset(args.repo_name, args.base_path)


if __name__ == "__main__":
    main()

