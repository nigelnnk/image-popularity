from abc import ABC, abstractmethod

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class BaseTrainer(ABC):
    def __init__(
            self,
            data_loader_train,
            data_loader_eval,
            model,
            num_epochs=100,
            steps_per_log=1,
            epochs_per_eval=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            weight_decay=0,
            log_dir=None,
            save_path=None):
        self.data_loader_train = data_loader_train
        self.data_loader_eval = data_loader_eval
        self.model = model

        self.num_epochs = num_epochs
        self.steps_per_log = steps_per_log
        self.epochs_per_eval = epochs_per_eval

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.log_dir = log_dir
        self.save_path = save_path

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        parameter_dicts = self.model.parameter_dicts
        self.optimizer = optim.AdamW(
            parameter_dicts,
            lr=self.learning_rate,
            weight_decay=self.weight_decay)

        if log_dir is not None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None

        self.epoch = 1
        self.global_step = 1

    def train(self):
        pbar = tqdm(
            range(self.epoch, self.num_epochs + 1),
            desc=f"Epoch: {self.epoch}")
        for epoch in pbar:
            self.epoch = epoch
            pbar.set_description(f"Epoch: {self.epoch}")

            self.model = self.model.train()
            self.train_epoch()

            if epoch % self.epochs_per_eval == 0:
                with torch.no_grad():
                    self.model = self.model.eval()
                    self.eval()

                self.save_model()

    def train_epoch(self):
        pbar = tqdm(self.data_loader_train, desc="Training Loss: ?")
        for step, data in enumerate(pbar, start=1):
            outputs, loss = self.train_forward(data)

            loss_normalized = loss / self.gradient_accumulation_steps
            loss_normalized.backward()

            if step % self.gradient_accumulation_steps == 0 \
                    or step >= len(self.data_loader_train):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if self.global_step % self.steps_per_log == 0:
                self.train_log(outputs, loss)
                pbar.set_description(f"Training Loss: {loss.item():.3g}")

            self.global_step += 1

    @abstractmethod
    def train_forward(self, data):
        pass

    @abstractmethod
    def train_log(self, outputs, loss):
        pass

    def eval(self):
        outputs_list = []
        labels_list = []

        pbar = tqdm(self.data_loader_eval, desc="Evaluation")
        for step, data in enumerate(pbar, start=1):
            outputs, labels = self.eval_forward(data)

            outputs_list.append(outputs)
            labels_list.append(labels)
        outputs = torch.cat(outputs_list)
        labels = torch.cat(labels_list)

        self.eval_log(outputs, labels)

    @abstractmethod
    def eval_forward(self, data):
        pass

    @abstractmethod
    def eval_log(self, outputs, labels):
        pass

    @abstractmethod
    def save_model(self):
        pass
