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
            reddit_level='multireddit',
            use_reddit_scores=False,
            filter=None,
            filter_labels=True,
            hierarchical=False,
            coarse_level='network',
            split='train',
            load_files_into_memory=False):
        self.path = path
        self.labels_path = labels_path
        self.image_size = image_size
        self.reddit_level = reddit_level
        self.use_reddit_scores = use_reddit_scores
        self.filter = filter
        self.filter_labels = filter_labels
        self.hierarchical = hierarchical
        self.coarse_level = coarse_level
        self.split = split
        self.load_files_into_memory = load_files_into_memory

        self.data = pd.read_csv(path)
        self.data = self.data[self.data['SPLIT'] == self.split]

        self.data = self.data[self.data[self.reddit_level.upper()].notna()]

        if self.filter:
            col, fil = self.filter.split(':')
            self.data = self.data[self.data[col.upper()] == fil]
            assert self.data.size > 0, 'Empty dataset; check filter?'

        self.data['REDDIT SCORE'] = (
            self.data[self.reddit_level.upper()]
            + '_'
            + self.data['PERCENTILE BIN'].astype(str))

        if self.load_files_into_memory:
            self.image_files = [None] * len(self.data)

        # Calculate sample weights to balance data during training
        class_count = self.data['REDDIT SCORE'].value_counts()
        self._sample_weights = [
            1.0 / class_count[score]
            for score in self.data['REDDIT SCORE']]

        with open(labels_path) as file:
            labels = json.load(file)

        self._subreddits = labels['subreddit']
        self._multireddits = labels['multireddit']
        self._networks = labels['network']

        if self.filter_labels and self.filter:
            fil = set(self.data['SUBREDDIT'].unique().tolist())
            self._subreddits = [x for x in self._subreddits if x in fil]

            fil = set(self.data['MULTIREDDIT'].unique().tolist())
            self._multireddits = [x for x in self._multireddits if x in fil]

            fil = set(self.data['NETWORK'].unique().tolist())
            self._networks = [x for x in self._networks if x in fil]

        self._reddit_scores = []
        percentile_bins = labels['percentile_bin']
        if self.reddit_level == 'subreddit':
            labels = self._subreddits
        elif self.reddit_level == 'multireddit':
            labels = self._multireddits
        elif self.reddit_level == 'network':
            labels = self._networks
        for label in labels:
            for score in percentile_bins:
                self._reddit_scores.append(f'{label}_{score}')

        self._id_to_subreddit = {
            k: v for k, v in enumerate(self._subreddits)}
        self._id_to_multireddit = {
            k: v for k, v in enumerate(self._multireddits)}
        self._id_to_network = {
            k: v for k, v in enumerate(self._networks)}
        self._id_to_reddit_score = {
            k: v for k, v in enumerate(self._reddit_scores)}

        self._subreddit_to_id = {
            v: k for k, v in enumerate(self._subreddits)}
        self._multireddit_to_id = {
            v: k for k, v in enumerate(self._multireddits)}
        self._network_to_id = {
            v: k for k, v in enumerate(self._networks)}
        self._reddit_score_to_id = {
            v: k for k, v in enumerate(self._reddit_scores)}

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
                # Analysis says that only 2% of the images will occupy less
                # than half the image
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

        if self.use_reddit_scores:
            label = self.label_to_id(data['REDDIT SCORE'])
        else:
            label = self.label_to_id(data[self.reddit_level.upper()])

        if self.hierarchical:
            reddit_level = self.reddit_level
            use_reddit_scores = self.use_reddit_scores

            self.reddit_level = self.coarse_level
            self.use_reddit_scores = False
            coarse_label = self.label_to_id(data[self.reddit_level.upper()])

            self.reddit_level = reddit_level
            self.use_reddit_scores = use_reddit_scores

            return image, (label, coarse_label)
        else:
            return image, label

    @property
    def transforms(self):
        return self._transforms

    @property
    def sample_weights(self):
        return self._sample_weights

    @property
    def labels(self):
        if self.use_reddit_scores:
            return self._reddit_scores
        elif self.reddit_level == 'subreddit':
            return self._subreddits
        elif self.reddit_level == 'multireddit':
            return self._multireddits
        elif self.reddit_level == 'network':
            return self._networks

    @property
    def subreddits(self):
        return self._subreddits

    @property
    def multireddits(self):
        return self._multireddits

    @property
    def networks(self):
        return self._networks

    @property
    def reddit_scores(self):
        return self._reddit_scores

    def id_to_label(self, id):
        if self.use_reddit_scores:
            return self._id_to_reddit_score[id]
        elif self.reddit_level == 'subreddit':
            return self._id_to_subreddit[id]
        elif self.reddit_level == 'multireddit':
            return self._id_to_multireddit[id]
        elif self.reddit_level == 'network':
            return self._id_to_network[id]

    def id_to_subreddit(self, id):
        return self._id_to_subreddit[id]

    def id_to_multireddit(self, id):
        return self._id_to_multireddit[id]

    def id_to_network(self, id):
        return self._id_to_network[id]

    def id_to_reddit_score(self, id):
        return self._id_to_reddit_score[id]

    def label_to_id(self, label):
        if self.use_reddit_scores:
            return self._reddit_score_to_id[label]
        elif self.reddit_level == 'subreddit':
            return self._subreddit_to_id[label]
        elif self.reddit_level == 'multireddit':
            return self._multireddit_to_id[label]
        elif self.reddit_level == 'network':
            return self._network_to_id[label]

    def subreddit_to_id(self, subreddit):
        return self._subreddit_to_id[subreddit]

    def multireddit_to_id(self, multireddit):
        return self._multireddit_to_id[multireddit]

    def network_to_id(self, network):
        return self._network_to_id[network]

    def reddit_score_to_id(self, reddit_score):
        return self._reddit_score_to_id[reddit_score]
