import random

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from tqdm import tqdm

from train.classify import classify
from train.load import load_data, load_dataset, load_model, load_trained_model
from train.trainer import Trainer

CONFIG = {
    'data_path': 'data/reddit_data.csv',
    'labels_path': 'data/reddit_labels.json',

    'save_path': 'data/models/efficientnet',
    'log_dir': 'data/runs/efficientnet',

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
    'model_name': 'efficientnet_b0',
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
        image_size=(224, 224),
        model_name='efficientnet_b0',
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
        filter=None)

    # Train multireddit classifier

    data_loader_train = load_data(
        data_path,
        labels_path,
        'multireddit',
        'train',
        image_size,
        use_reddit_scores=False,
        filter=None,
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
        use_reddit_scores=False,
        filter=None,
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
        log_dir=f'{log_dir}_multireddit',
        save_path=f'{save_path}_multireddit')

    trainer.train()

    # Train multireddit score classifiers

    for multireddit in dataset.multireddits:
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        data_loader_train = load_data(
            data_path,
            labels_path,
            'multireddit',
            'train',
            image_size,
            use_reddit_scores=True,
            filter=f'multireddit:{multireddit}',
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
            filter=f'multireddit:{multireddit}',
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
            log_dir=f'{log_dir}_{multireddit}_score',
            save_path=f'{save_path}_{multireddit}_score')

        trainer.train()


def evaluate(
        data_path,
        labels_path,
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

    dataset = load_dataset(
        data_path,
        labels_path,
        'multireddit',
        'test',
        image_size,
        use_reddit_scores=True,
        filter=None)

    # Evaluate multireddit classifier
    data_loader_eval = load_data(
        data_path,
        labels_path,
        'multireddit',
        'test',
        image_size,
        use_reddit_scores=False,
        filter=None,
        load_files_into_memory=load_files_into_memory,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor)
    model = load_trained_model(model_name, f'{save_path}_multireddit')
    model = model.eval()

    multireddit_outputs, multireddit_labels = classify(data_loader_eval, model)

    print('Multireddit Classification')
    print(classification_report(
        multireddit_labels.detach().cpu().numpy(),
        torch.argmax(multireddit_outputs, dim=-1).detach().cpu().numpy(),
        labels=range(len(dataset.multireddits)),
        target_names=dataset.multireddits,
        digits=5,
        zero_division=0))

    # Evaluate multireddit score classifiers
    data_loader_eval = load_data(
        data_path,
        labels_path,
        'multireddit',
        'test',
        image_size,
        use_reddit_scores=True,
        filter=None,
        load_files_into_memory=load_files_into_memory,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor)

    scores_outputs_list = []
    pbar = tqdm(dataset.multireddits)
    for multireddit in pbar:
        pbar.set_description(multireddit)
        model = load_trained_model(
            model_name, f'{save_path}_{multireddit}_score')
        model = model.eval()

        scores_outputs, scores_labels = classify(data_loader_eval, model)
        scores_outputs_list.append(scores_outputs)
    scores_outputs = torch.stack(scores_outputs_list, 1)

    # Evaluation assuming perfect multireddit classifier
    perfect_multireddit = F.one_hot(multireddit_labels)
    perfect_multireddit = torch.unsqueeze(perfect_multireddit, dim=-1)
    perfect_scores_outputs = perfect_multireddit * scores_outputs
    perfect_scores_outputs = torch.reshape(
        perfect_scores_outputs,  (perfect_scores_outputs.shape[0], -1))

    print('Multireddit Score Classification (Perfect Multireddit Classifier)')
    print(classification_report(
        scores_labels.detach().cpu().numpy(),
        torch.argmax(perfect_scores_outputs, dim=-1).detach().cpu().numpy(),
        labels=range(len(dataset.labels)),
        target_names=dataset.labels,
        digits=5,
        zero_division=0))

    # Evaluation assuming trained multireddit classifier
    multireddit_outputs = torch.unsqueeze(multireddit_outputs, dim=-1)
    scores_outputs = multireddit_outputs * scores_outputs
    scores_outputs = torch.reshape(
        scores_outputs,  (scores_outputs.shape[0], -1))

    print('Multireddit Score Classification (Trained Multireddit Classifier)')
    print(classification_report(
        scores_labels.detach().cpu().numpy(),
        torch.argmax(scores_outputs, dim=-1).detach().cpu().numpy(),
        labels=range(len(dataset.labels)),
        target_names=dataset.labels,
        digits=5,
        zero_division=0))


if __name__ == '__main__':
    train(**CONFIG)
    evaluate(**CONFIG)
