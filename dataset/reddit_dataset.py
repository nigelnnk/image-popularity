import json

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm import tqdm


class RedditDataset(Dataset):
    def __init__(
            self,
            path,
            label_map_path,
            image_size,
            split='train',
            load_files_into_memory=False):
        self.path = path
        self.label_map_path = label_map_path
        self.image_size = image_size
        self.split = split
        self.load_files_into_memory = load_files_into_memory

        self.data = pd.read_csv(path)
        self.data = self.data[self.data['SPLIT'] == self.split]

        # Load files into memory to minimise disk reads
        if self.load_files_into_memory:
            print(f'Reading {len(self.data)} image files into memory')
            self.image_files = []
            for path in tqdm(self.data['PATH']):
                self.image_files.append(
                    torch.from_numpy(np.fromfile(path, dtype=np.uint8)))

        # Calculate sample weights to balance data during training
        class_count = self.data['PERCENTILE BIN'].value_counts()
        self._sample_weights = [
            1.0 / class_count[percentile_bin]
            for percentile_bin in self.data['PERCENTILE BIN']]

        with open(label_map_path) as file:
            label_map = json.load(file)
        self._id_to_subreddit = label_map['subreddit_map']
        self._id_to_percentile_bin = label_map['percentile_bin_map']
        self._subreddit_to_id = {
            v: k for k, v in self._id_to_subreddit.items()}
        self._percentile_bin_to_id = {
            v: k for k, v in self._id_to_percentile_bin.items()}

        # Data augmentations and transforms
        if self.split == 'train':
            self.resize_crop = torch.nn.Sequential(
                transforms.RandomResizedCrop(
                    image_size, scale=(0.08, 1), ratio=(3 / 4, 4 / 3)),
            )
            self.transforms = torch.nn.Sequential(
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
                transforms.Resize((max(image_size), max(image_size))),
                transforms.CenterCrop(image_size),
            )
            self.transforms = torch.nn.Sequential(
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
            image_file = self.image_files[idx]
            image = torchvision.io.decode_image(image_file)
        else:
            path = data['PATH']
            image = torchvision.io.read_image(path)
        image = image / 255.0
        image = self.resize_crop(image)
        image = self.transforms(image)

        subreddit = self.subreddit_to_id(data['SUBREDDIT'])
        percentile_bin = self.percentile_bin_to_id(data['PERCENTILE BIN'])

        return image, subreddit, percentile_bin

    @property
    def sample_weights(self):
        return self._sample_weights

    def id_to_subreddit(self, id):
        return self._id_to_subreddit[id]

    def id_to_percentile_bin(self, id):
        return self._id_to_percentile_bin[id]

    def subreddit_to_id(self, subreddit):
        return self._subreddit_to_id[subreddit]

    def percentile_bin_to_id(self, percentile_bin):
        return self._percentile_bin_to_id[percentile_bin]
