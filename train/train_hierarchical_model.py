import random

import torch
from sklearn.metrics import classification_report

from train.classify import classify
from train.load import (load_data, load_dataset, load_hierarchical_model,
                        load_trained_model)
from train.trainer import Trainer

CONFIG = {
    'data_path': 'data/reddit_data.csv',
    'labels_path': 'data/reddit_labels.json',

    'save_path': 'data/models/hierarchical',
    'log_dir': 'data/runs/hierarchical',

    'num_epochs': 10,
    'steps_per_log': 25,
    'epochs_per_eval': 10,

    'gradient_accumulation_steps': 1,
    'batch_size': 128,
    'num_workers': 4,
    'prefetch_factor': 4,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,

    'image_size': (224, 224),
    'efficientnet_model_name': 'efficientnet_b0',
    'dropout_rate': 0.2,

    # NOTE: Not recommended, each worker will load all image files into memory
    'load_files_into_memory': False,

    'random_seed': 0,
}


def train(
        data_path,
        labels_path,
        save_path,
        log_dir=None,
        num_epochs=10,
        steps_per_log=100,
        epochs_per_eval=1,
        gradient_accumulation_steps=1,
        batch_size=32,
        num_workers=8,
        prefetch_factor=4,
        learning_rate=1e-3,
        weight_decay=1e-5,
        image_size=(256, 256),
        efficientnet_model_name='efficientnet_b0',
        dropout_rate=0.1,
        load_files_into_memory=False,
        random_seed=0):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    # torch.use_deterministic_algorithms(True, warn_only=True)

    dataset = load_dataset(
        data_path,
        labels_path,
        'multireddit',
        'train',
        image_size,
        use_reddit_scores=True,
        filter=None,
        hierarchical=True,
        coarse_level='multireddit')

    model = load_hierarchical_model(
        efficientnet_model_name,
        len(dataset.multireddits),
        int(len(dataset.reddit_scores) / len(dataset.multireddits)),
        dropout_rate=dropout_rate)

    data_loader_train = load_data(
        data_path,
        labels_path,
        'multireddit',
        'train',
        image_size,
        use_reddit_scores=True,
        filter=None,
        hierarchical=True,
        coarse_level='multireddit',
        load_files_into_memory=load_files_into_memory,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor)

    data_loader_eval = load_data(
        data_path,
        labels_path,
        'multireddit',
        'val',
        image_size,
        use_reddit_scores=True,
        filter=None,
        hierarchical=False,
        coarse_level='multireddit',
        load_files_into_memory=load_files_into_memory,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor)

    trainer = Trainer(
        data_loader_train,
        data_loader_eval,
        model,
        num_epochs=num_epochs,
        steps_per_log=steps_per_log,
        epochs_per_eval=epochs_per_eval,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        log_dir=log_dir,
        save_path=save_path)
    trainer.train()


def evaluate(
        data_path,
        labels_path,
        save_path,
        batch_size=32,
        num_workers=8,
        prefetch_factor=4,
        image_size=(256, 256),
        load_files_into_memory=False,
        random_seed=0,
        **kwargs):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    dataset = load_dataset(
        data_path,
        labels_path,
        'multireddit',
        'test',
        image_size,
        use_reddit_scores=True,
        filter=None,
        hierarchical=True,
        coarse_level='multireddit')

    data_loader_eval = load_data(
        data_path,
        labels_path,
        'multireddit',
        'test',
        image_size,
        use_reddit_scores=True,
        filter=None,
        hierarchical=True,
        coarse_level='multireddit',
        load_files_into_memory=load_files_into_memory,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor)

    model = load_trained_model('hierarchical', save_path)
    model = model.eval()

    outputs, labels = classify(data_loader_eval, model)

    print('Multireddit Classification')
    print(classification_report(
        labels[1].detach().cpu().numpy(),
        torch.argmax(outputs[1], dim=-1).detach().cpu().numpy(),
        labels=range(len(dataset.multireddits)),
        target_names=dataset.multireddits,
        digits=5,
        zero_division=0))

    print('Multireddit Score Classification')
    print(classification_report(
        labels[0].detach().cpu().numpy(),
        torch.argmax(outputs[0], dim=-1).detach().cpu().numpy(),
        labels=range(len(dataset.labels)),
        target_names=dataset.labels,
        digits=5,
        zero_division=0))


if __name__ == '__main__':
    train(**CONFIG)
    evaluate(**CONFIG)
