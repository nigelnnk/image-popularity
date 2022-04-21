import random

import optuna
import torch
from torch.utils.data import BatchSampler, DataLoader, WeightedRandomSampler

from dataset.reddit_dataset import RedditDataset
from model.dummy_model import DummyModel
from model.efficientnet import EfficientNet
from train.trainer import Trainer

CONFIG = {
    'data_path': 'data/short_reddit_data.csv',
    'labels_path': 'data/reddit_labels.json',

    # Must be one of ['subreddit', 'multireddit', 'network']
    'reddit_level': 'multireddit',
    'use_reddit_scores': True,
    'filter': None,

    'save_path': 'data/models/efficientnet',
    'log_dir': 'data/runs/efficientnet',

    'num_epochs': 10,
    'steps_per_log': 100,
    'epochs_per_eval': 1,

    'gradient_accumulation_steps': 1,
    'batch_size': 128,
    'num_workers': 8,
    'prefetch_factor': 4,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,

    'image_size': (224, 224),
    'model_name': 'efficientnet_b0',
    'dropout_rate': 0.2,

    # NOTE: Not recommended, each worker will load all image files into memory
    'load_files_into_memory': False,

    'random_seed': 0,
}


def load_data(
        data_path,
        labels_path,
        reddit_level,
        use_reddit_scores,
        filter,
        split,
        image_size,
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
            prefetch_factor=prefetch_factor,
            persistent_workers=True)
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=True)

    return data_loader


def load_model(model_name, num_outputs, dropout_rate=0.1):
    if model_name == 'dummy':
        model = DummyModel(3, 256, num_outputs)
    elif 'efficientnet' in model_name:
        model = EfficientNet(
            model_name,
            num_outputs,
            dropout_rate=dropout_rate,
            efficientnet_pretrained=True)
    return model


def train(
        data_path,
        labels_path,
        reddit_level,
        use_reddit_scores,
        filter,
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
        image_size=(224, 224),
        model_name='efficientnet_b0',
        dropout_rate=0.1,
        load_files_into_memory=False,
        random_seed=0):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.use_deterministic_algorithms(True)

    data_loader_train = load_data(
        data_path,
        labels_path,
        reddit_level,
        use_reddit_scores,
        filter,
        'train',
        image_size,
        load_files_into_memory=load_files_into_memory,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor)

    data_loader_eval = load_data(
        data_path,
        labels_path,
        reddit_level,
        use_reddit_scores,
        filter,
        'val',
        image_size,
        load_files_into_memory=load_files_into_memory,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor)

    model = load_model(
        model_name,
        len(data_loader_train.dataset.labels),
        dropout_rate=dropout_rate)

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
    return trainer.best_f1_score


def tune_hyperparameters(**config):
    def objective(trial):
        config['num_epochs'] = 10
        config['epochs_per_eval'] = 1

        config['learning_rate'] = trial.suggest_float(
            'learning_rate', 1e-5, 1e-2, log=True)
        config['weight_decay'] = trial.suggest_float(
            'weight_decay', 1e-5, 1e-2, log=True)

        return train(**config)

    def print_status(study, trial):
        print(f"Trial Params:\n {trial.params}")
        print(f"Trial Value:\n {trial.value}")
        print(f"Best Params:\n {study.best_params}")
        print(f"Best Value:\n {study.best_value}")

    study = optuna.create_study(study_name='dummy', direction='maximize')
    study.optimize(objective, n_trials=100, callbacks=[print_status])
    print(study.best_params)
    print(study.best_value)


if __name__ == '__main__':
    # tune_hyperparameters(**CONFIG)
    train(**CONFIG)
