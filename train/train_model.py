import random

import torch
from sklearn.metrics import classification_report

from train.classify import classify
from train.load import load_data, load_model, load_trained_model
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
    'epochs_per_eval': 10,

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
    torch.use_deterministic_algorithms(True, warn_only=True)

    data_loader_train = load_data(
        data_path,
        labels_path,
        reddit_level,
        'train',
        image_size,
        use_reddit_scores=use_reddit_scores,
        filter=filter,
        load_files_into_memory=load_files_into_memory,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor)

    data_loader_eval = load_data(
        data_path,
        labels_path,
        reddit_level,
        'val',
        image_size,
        use_reddit_scores=use_reddit_scores,
        filter=filter,
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


def evaluate(
        data_path,
        labels_path,
        reddit_level,
        use_reddit_scores,
        filter,
        save_path,
        batch_size=32,
        num_workers=8,
        prefetch_factor=4,
        image_size=(224, 224),
        model_name='efficientnet_b0',
        load_files_into_memory=False,
        random_seed=0,
        **kwargs):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    data_loader_eval = load_data(
        data_path,
        labels_path,
        reddit_level,
        'test',
        image_size,
        use_reddit_scores=use_reddit_scores,
        filter=filter,
        load_files_into_memory=load_files_into_memory,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor)

    model = load_trained_model(model_name, save_path)
    model = model.eval()

    outputs, labels = classify(data_loader_eval, model)

    print('Multireddit Score Classification')
    print(classification_report(
        labels.detach().cpu().numpy(),
        torch.argmax(outputs, dim=-1).detach().cpu().numpy(),
        labels=range(len(data_loader_eval.dataset.labels)),
        target_names=data_loader_eval.dataset.labels,
        digits=5,
        zero_division=0))


if __name__ == '__main__':
    train(**CONFIG)
    evaluate(**CONFIG)
