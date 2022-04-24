import random

import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import classification_report

from dataset.reddit_dataset import RedditDataset
from model.alexnet import AlexNet
from model.dummy_model import DummyModel
from model.efficientnet import EfficientNet
from train.trainer import Trainer
from train.utils import recursive_to_device


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    'data_path': 'data/short_reddit_data.csv',
    'labels_path': 'data/reddit_labels.json',

    'save_path': 'data/models/alexnet',
    'log_dir': 'data/runs/alexnet',

    'num_epochs': 5,
    'steps_per_log': 100,
    'epochs_per_eval': 1,

    'gradient_accumulation_steps': 1,
    'batch_size': 128,
    'num_workers': 8,
    'prefetch_factor': 4,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,

    'image_size': (224, 224),
    'model_name': 'alexnet',
    'dropout_rate': 0.2,

    # NOTE: Not recommended, each worker will load all image files into memory
    'load_files_into_memory': False,

    'random_seed': 0,
}


def load_dataset(
        data_path,
        labels_path,
        reddit_level,
        split,
        image_size,
        use_reddit_scores=True,
        filter=None):
    dataset = RedditDataset(
        data_path,
        labels_path,
        image_size,
        reddit_level=reddit_level,
        use_reddit_scores=use_reddit_scores,
        filter=filter,
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
    elif model_name == 'alexnet':
        model = AlexNet(num_outputs, use_pretrained=True)
    elif 'efficientnet' in model_name:
        model = EfficientNet(
            model_name,
            num_outputs,
            dropout_rate=dropout_rate,
            efficientnet_pretrained=True)
    return model


def load_trained_model(model_name, model_path):
    if model_name == 'dummy':
        model = DummyModel.load(model_path)
    elif model_name == 'alexnet':
        model = AlexNet.load(model_path)
    elif 'efficientnet' in model_name:
        model = EfficientNet.load(model_path)
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
        image_size=(224, 224),
        model_name='efficientnet_b0',
        dropout_rate=0.1,
        load_files_into_memory=False,
        random_seed=0):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

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

    # Evaluate multireddit classifier
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
    model = load_trained_model(model_name, f'{save_path}_multireddit')
    model = model.to(device)
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
        'val',
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
        model = model.to(device)
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


def classify(data_loader, model):
    with torch.no_grad():
        outputs_list = []
        labels_list = []
        for data in tqdm(data_loader, leave=False):
            data = recursive_to_device(data, device, non_blocking=True)
            images, labels = data
            images = data_loader.dataset.transforms(images)

            outputs = F.softmax(model(images), dim=-1)

            outputs_list.append(outputs)
            labels_list.append(labels)
        outputs = torch.cat(outputs_list)
        labels = torch.cat(labels_list)
        return outputs, labels


if __name__ == '__main__':
    train(**CONFIG)