#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset.utils import split_dataset

PERCENTILE_BINS = [0.25, 0.5, 0.75, 0.9, 1.0]


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Collate reddit data into single CSV file.")

    parser.add_argument(
        '--data_path', default='data/reddit', type=str,
        help='Path to reddit data.')

    parser.add_argument(
        '--reddit_levels_path', default='dataset/reddit_levels.csv', type=str,
        help='Path to reddit levels.')

    parser.add_argument(
        '--output_path', default='data/new_reddit_data.csv', type=str,
        help='Path to output collated data CSV file.')

    parser.add_argument(
        '--labels_path', default='data/new_reddit_labels.json', type=str,
        help='Path to output labels.')

    parser.add_argument(
        '--min_posts', default=500, type=int,
        help='Minimum number of posts in each subreddit.')

    parser.add_argument(
        '--train_size', default=0.8, type=float,
        help='Ratio of data to use for train set.')

    parser.add_argument(
        '--val_size', default=0.1, type=float,
        help="""Ratio of data to use for validation set.
                Remaining will be used for test set.""")

    return parser


def collate_reddit_data(
        data_path,
        reddit_levels_path,
        output_path,
        labels_path,
        min_posts=1,
        train_size=0.8,
        val_size=0.1):
    data_path = Path(data_path)
    print(f'Reading data from: {data_path}')

    csv_paths = list(data_path.glob('**/*.csv'))
    csv_paths.sort()

    image_paths = data_path.glob('**/*.jpeg')
    image_id_to_path = {path.stem: path for path in image_paths}

    # Load individual subreddit data
    skipped_subreddits = []
    data_list = []
    for csv_path in tqdm(csv_paths):
        data = pd.read_csv(csv_path, skiprows=2, on_bad_lines='skip')
        data = data[data['ID'].isin(image_id_to_path)]
        if len(data) < min_posts:
            skipped_subreddits.append(data['SUBREDDIT'][0])
            continue

        data['PATH'] = data['ID'].map(image_id_to_path)

        # Get percentile and percentile bin for each post in subreddit
        data['PERCENTILE'] = data['SCORE'].rank(pct=True)
        data['PERCENTILE BIN'] = np.digitize(
            data['PERCENTILE'], PERCENTILE_BINS, right=True)
        data['PERCENTILE BIN'] = data['PERCENTILE BIN'].map(
            {index: bin for index, bin in enumerate(PERCENTILE_BINS)})

        data_list.append(data)
    data = pd.concat(data_list, ignore_index=True)
    print(f'Skipped subreddits: {skipped_subreddits}')

    # Merge reddit levels
    reddit_levels = pd.read_csv(reddit_levels_path)
    data = pd.merge(data, reddit_levels, how='left', on='SUBREDDIT')
    data["LOG SCORE BIN"] = np.floor(np.log10(data["SCORE"])).clip(0, 6)

    # Create and save labels
    labels = {
        'percentile_bin': PERCENTILE_BINS,
        "log_score_bin": list(range(data["LOG SCORE BIN"].to_numpy().max().astype(int) + 1))
    }
    for level in reddit_levels:
        labels[level.lower()] = list(data[level].dropna().unique())
    with open(labels_path, 'w') as file:
        json.dump(labels, file, indent=4)
    print(f'Saved labels to {labels_path}')

    # Split dataset
    train, val, test = split_dataset(
        data,
        train_size,
        val_size,
        stratify_by='PERCENTILE BIN',
        random_seed=0)
    train['SPLIT'] = 'train'
    val['SPLIT'] = 'val'
    test['SPLIT'] = 'test'
    data = pd.concat([train, val, test], ignore_index=True)

    data.to_csv(output_path, index=False)
    print(f'Saved data to {output_path}')


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    collate_reddit_data(**vars(args))
