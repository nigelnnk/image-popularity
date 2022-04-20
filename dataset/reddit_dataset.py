import json

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class RedditDataset(Dataset):
    def __init__(
            self,
            path,
            labels_path,
            image_size,
            reddit_level='subreddit',
            split='train',
            score_transform="log",
            load_files_into_memory=False):
        self.path = path
        self.labels_path = labels_path
        self.image_size = image_size
        self.reddit_level = reddit_level
        self.split = split
        self.load_files_into_memory = load_files_into_memory
        score_column = {"log": "LOG SCORE BIN", "percentile": "PERCENTILE BIN"}
        self.score_transform = score_column[score_transform]

        self.data = pd.read_csv(path)
        self.data = self.data[self.data['SPLIT'] == self.split]

        self.data = self.data[self.data[self.reddit_level.upper()].notna()]

        if self.load_files_into_memory:
            self.image_files = [None] * len(self.data)

        # Calculate sample weights to balance data during training
        self.data['FULL CLASS'] = (
            self.data[self.reddit_level.upper()]
            + self.data[self.score_transform].astype(str))
        class_count = self.data['FULL CLASS'].value_counts()
        self._sample_weights = [
            1.0 / class_count[full_class]
            for full_class in self.data['FULL CLASS']]

        with open(labels_path) as file:
            labels = json.load(file)
        self._subreddits = labels[self.reddit_level]
        self._percentile_bins = labels['percentile_bin']
        self._log_score_bins = labels["log_score_bin"]
        self._id_to_subreddit = {
            k: v for k, v in enumerate(self._subreddits)}
        self._id_to_percentile_bin = {
            k: v for k, v in enumerate(self._percentile_bins)}
        self._id_to_log_score_bin = {
            k: v for k, v in enumerate(self._log_score_bins)}
        self._subreddit_to_id = {
            v: k for k, v in enumerate(self._subreddits)}
        self._percentile_bin_to_id = {
            v: k for k, v in enumerate(self._percentile_bins)}
        self._log_score_bin_to_id = {
            v: k for k, v in enumerate(self._log_score_bins)}

        # Data augmentations and transforms
        if self.split == 'train':
            self.resize_crop = torch.nn.Sequential(
                transforms.RandomResizedCrop(
                    image_size, scale=(0.08, 1), ratio=(3 / 4, 4 / 3)),
            )
            self._transforms = torch.nn.Sequential(
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # See https://pytorch.org/vision/stable/models.html for values
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
            )
        else:
            self.resize_crop = torch.nn.Sequential(
                # Analysis says that only 2% of the images will occupy less than half the image
                transforms.Resize(max(image_size)),
                transforms.CenterCrop(image_size),
            )
            self._transforms = torch.nn.Sequential(
                # See https://pytorch.org/vision/stable/models.html for values
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx]

        if self.load_files_into_memory:
            # Load files into memory to minimise disk reads
            if self.image_files[idx] is None:
                self.image_files[idx] = torch.from_numpy(np.fromfile(
                    data['PATH'], dtype=np.uint8))

            image_file = self.image_files[idx]
            image = torchvision.io.decode_image(image_file)
        else:
            image = torchvision.io.read_image(data['PATH'])
        image = image / 255.0
        image = self.resize_crop(image)

        subreddit = self.subreddit_to_id(data[self.reddit_level.upper()])
        if self.score_transform == "PERCENTILE BIN":
            bin = self.percentile_bin_to_id(data['PERCENTILE BIN'])
        elif self.score_transform == "LOG SCORE BIN":
            bin = self.log_score_bin_to_id(data['LOG SCORE BIN'])

        return image, subreddit, bin

    @property
    def transforms(self):
        return self._transforms

    @property
    def sample_weights(self):
        return self._sample_weights

    @property
    def subreddits(self):
        return self._subreddits

    @property
    def percentile_bins(self):
        return self._percentile_bins

    @property
    def log_score_bins(self):
        return self._log_score_bins

    def id_to_subreddit(self, id):
        return self._id_to_subreddit[id]

    def id_to_percentile_bin(self, id):
        return self._id_to_percentile_bin[id]
    
    def id_to_log_score_bin(self, id):
        return self._id_to_log_score_bin[id]

    def subreddit_to_id(self, subreddit):
        return self._subreddit_to_id[subreddit]

    def percentile_bin_to_id(self, percentile_bin):
        return self._percentile_bin_to_id[percentile_bin]

    def log_score_bin_to_id(self, bin):
        return self._log_score_bin_to_id[bin]
