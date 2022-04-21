import argparse
import random
import optuna
import torch
from torch.utils.data import BatchSampler, DataLoader, WeightedRandomSampler

from dataset.reddit_dataset import RedditDataset
from model.alexnet import AlexNet
from train.alexnet_trainer import AlexNet_Trainer

CONFIG = {
    # AlexNet specifics at bottom of code

    'data_path': 'data/new_reddit_data.csv',
    'labels_path': 'data/new_reddit_labels.json',

    'save_path': 'data/models/alexnet_mix',
    'log_dir': 'data/runs/alexnet_mix',

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
        reddit_level='subreddit',
        score_transform="log",
        load_files_into_memory=False,
        batch_size=32,
        num_workers=8,
        prefetch_factor=4):
    dataset = RedditDataset(
        data_path,
        labels_path,
        image_size,
        reddit_level=reddit_level,
        split=split,
        score_transform=score_transform,
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


def load_model(input_channels, output_channels, use_pretrained=False, feature_extract=False):
    model = AlexNet(input_channels, output_channels, use_pretrained, feature_extract)
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
        feature_extract=True,
        use_pretrained=True,
        target="subreddits",
        overfit=False):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.use_deterministic_algorithms(True)

    if target in ["multireddit" , "network"]:
        reddit_level = target
    elif target == "mix":
        reddit_level = "multireddit"
    else:
        reddit_level = "subreddit"

    if target in ["log"]:
        score_transform = "log"
    else:
        score_transform = "percentile"

    data_loader_train = load_data(
        data_path,
        labels_path,
        'train',
        image_size,
        reddit_level=reddit_level,
        score_transform=score_transform,
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
            reddit_level=reddit_level,
            score_transform=score_transform,
            load_files_into_memory=load_files_into_memory,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor)

    output_dict = {'percentile': 3, 'log': 3, 'subreddit': 134, 'multireddit': 11, 'network': 2, 'mix': 3*11}
    output_len = output_dict[target]

    model = load_model(
        3, output_len, use_pretrained=use_pretrained, feature_extract=feature_extract)

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

def arg_parser():
    parser = argparse.ArgumentParser(
        description="Train AlexNet-like model on reddit datset")
    parser.add_argument(
        '--target', default='subreddit', type=str,
        help='Attribute to train on: percentile, subreddit, multireddit, network')
    parser.add_argument(
        '--pretrained', action='store_true',
        help='Use pretrained weights from torchvision or random weights')
    parser.add_argument(
        '--overfit', action='store_true',
        help='Set test set to be same as train set, used for debugging')
    parser.add_argument(
        '--feature_extract', action='store_true',
        help='Freeze nonfinal layers of model')
    return parser


if __name__ == '__main__':
    args = arg_parser().parse_args()
    CONFIG["target"] = args.target
    CONFIG["use_pretrained"] = args.pretrained
    CONFIG["overfit"] = args.overfit
    CONFIG["feature_extract"] = args.feature_extract
    
    # tune_hyperparameters(**CONFIG)
    train(**CONFIG)
