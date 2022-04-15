import random

import optuna
import torch
from torch.utils.data import BatchSampler, DataLoader, WeightedRandomSampler

from dataset.reddit_dataset import RedditDataset
from model.alexnet import AlexNet
from train.alexnet_trainer import AlexNet_Trainer

CONFIG = {
    # AlexNet specifics at bottom of code

    'data_path': 'data/reddit_data.csv',
    'labels_path': 'data/reddit_labels.json',

    'save_path': 'data/models/alexnet_finetune',
    'log_dir': 'data/runs/alexnet_finetune',

    'num_epochs': 10,
    'steps_per_log': 100,
    'epochs_per_eval': 1,

    'gradient_accumulation_steps': 1,
    'batch_size': 128,
    'num_workers': 8,
    'prefetch_factor': 4,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,

    'image_size': (227, 227),
    'hidden_channels': 256,

    # NOTE: Not recommended, each worker will load all image files into memory
    'load_files_into_memory': False,

    'random_seed': 0,
}


def load_data(
        data_path,
        labels_path,
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


def load_model(input_channels, hidden_channels, output_channels, use_pretrained=False):
    model = AlexNet(input_channels, hidden_channels, output_channels, use_pretrained)
    return model


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
        hidden_channels=32,
        load_files_into_memory=False,
        random_seed=0,
        use_pretrained=True,
        target="subreddits",
        overfit=False):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.use_deterministic_algorithms(True)

    data_loader_train = load_data(
        data_path,
        labels_path,
        'train',
        image_size,
        load_files_into_memory=load_files_into_memory,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor)

    if overfit:
        data_loader_eval = data_loader_train
    else:
        data_loader_eval = load_data(
            data_path,
            labels_path,
            'val',
            image_size,
            load_files_into_memory=load_files_into_memory,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor)

    if target == "percentiles":
        output_len = 5
    else:
        output_len = len(data_loader_train.dataset.subreddits)

    model = load_model(
        3, hidden_channels, output_len, use_pretrained=use_pretrained)

    trainer = AlexNet_Trainer(
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
        save_path=save_path,
        target=target)

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

    study = optuna.create_study(study_name='AlexNet', direction='maximize')
    study.optimize(objective, n_trials=100, callbacks=[print_status])
    print(study.best_params)
    print(study.best_value)


if __name__ == '__main__':
    CONFIG["target"] = "percentiles"
    CONFIG["use_pretrained"] = True
    CONFIG["overfit"] = True
    
    # tune_hyperparameters(**CONFIG)
    train(**CONFIG)
