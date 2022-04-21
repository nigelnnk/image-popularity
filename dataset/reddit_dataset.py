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
            filter="",
            load_files_into_memory=False):
        self.path = path
        self.labels_path = labels_path
        self.image_size = image_size
        self.reddit_level = reddit_level
        self.split = split
        self.load_files_into_memory = load_files_into_memory

        self.data = pd.read_csv(path)
        self.data = self.data[self.data['SPLIT'] == self.split]

        self.data = self.data[self.data[self.reddit_level.upper()].notna()]

        self.filter = filter
        if self.filter:
            col, fil = self.filter.split(":")
            self.data = self.data[self.data[col.upper()] == fil]
            assert self.data.size > 0, "Empty dataset; check filter?"

        if self.load_files_into_memory:
            self.image_files = [None] * len(self.data)

        # Calculate sample weights to balance data during training
        self.data['FULL CLASS'] = (
            self.data[self.reddit_level.upper()]
            + self.data["PERCENTILE BIN"].astype(str))
        class_count = self.data['FULL CLASS'].value_counts()
        self._sample_weights = [
            1.0 / class_count[full_class]
            for full_class in self.data['FULL CLASS']]

        with open(labels_path) as file:
            labels = json.load(file)
        self._subreddits = labels[self.reddit_level]
        if self.filter:
            in_data = self.data[self.reddit_level.upper()].unique().tolist()
            self._subreddits = [x for x in labels[self.reddit_level] if x in in_data]
        self._percentile_bins = labels['percentile_bin']
        
        self._id_to_subreddit = {
            k: v for k, v in enumerate(self._subreddits)}
        self._id_to_percentile_bin = {
            k: v for k, v in enumerate(self._percentile_bins)}
        
        self._subreddit_to_id = {
            v: k for k, v in enumerate(self._subreddits)}
        self._percentile_bin_to_id = {
            v: k for k, v in enumerate(self._percentile_bins)}

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
        bin = self.percentile_bin_to_id(data['PERCENTILE BIN'])

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

    def id_to_subreddit(self, id):
        return self._id_to_subreddit[id]

    def id_to_percentile_bin(self, id):
        return self._id_to_percentile_bin[id]

    def subreddit_to_id(self, subreddit):
        return self._subreddit_to_id[subreddit]

    def percentile_bin_to_id(self, percentile_bin):
        return self._percentile_bin_to_id[percentile_bin]
