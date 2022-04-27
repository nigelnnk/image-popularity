from dataset.reddit_dataset import RedditDataset
from model.alexnet import AlexNet
from model.efficientnet import EfficientNet
from model.hierarchical_model import HierarchicalEfficientNet
from torch.utils.data import BatchSampler, DataLoader, WeightedRandomSampler


def load_dataset(
        data_path,
        labels_path,
        reddit_level,
        split,
        image_size,
        use_reddit_scores=True,
        filter=None,
        hierarchical=False,
        coarse_level='multireddit'):
    dataset = RedditDataset(
        data_path,
        labels_path,
        image_size,
        reddit_level=reddit_level,
        use_reddit_scores=use_reddit_scores,
        filter=filter,
        hierarchical=hierarchical,
        coarse_level=coarse_level,
        split=split,
        load_files_into_memory=False)
    return dataset


def load_data(
        data_path,
        labels_path,
        reddit_level,
        split,
        image_size,
        use_reddit_scores=True,
        filter=None,
        hierarchical=False,
        coarse_level='multireddit',
        load_files_into_memory=False,
        batch_size=32,
        num_workers=8,
        prefetch_factor=4):
    dataset = RedditDataset(
        data_path,
        labels_path,
        image_size,
        reddit_level=reddit_level,
        use_reddit_scores=use_reddit_scores,
        filter=filter,
        hierarchical=hierarchical,
        coarse_level=coarse_level,
        split=split,
        load_files_into_memory=load_files_into_memory)

    if split == 'train':
        sampler = WeightedRandomSampler(
            dataset.sample_weights, len(dataset.sample_weights))
        batch_sampler = BatchSampler(sampler, batch_size, False)
        data_loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor)
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor)

    return data_loader


def load_model(model_name, num_outputs, dropout_rate=0.1):
    if model_name == 'alexnet':
        model = AlexNet(num_outputs, use_pretrained=True)
    elif 'efficientnet' in model_name:
        model = EfficientNet(
            model_name,
            num_outputs,
            dropout_rate=dropout_rate,
            efficientnet_pretrained=True)
    return model


def load_hierarchical_model(
        model_name,
        num_coarse_outputs,
        num_fine_outputs,
        dropout_rate=0.1):
    model = HierarchicalEfficientNet(
        model_name,
        num_coarse_outputs,
        num_fine_outputs,
        dropout_rate=dropout_rate,
        efficientnet_pretrained=True)
    return model


def load_trained_model(model_name, model_path):
    if model_name == 'alexnet':
        model = AlexNet.load(model_path)
    elif 'efficientnet' in model_name:
        model = EfficientNet.load(model_path)
    elif model_name == 'hierarchical':
        model = HierarchicalEfficientNet.load(model_path)
    return model
