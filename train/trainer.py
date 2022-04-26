import torch
import torch.cuda.amp as amp
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train.utils import recursive_to_device


class Trainer:
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

        self.cuda = True if torch.cuda.is_available() else False
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.model = self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=self.num_epochs)

        if self.cuda:
            self.scaler = amp.GradScaler()

        if log_dir is not None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None

        self.epoch = 1
        self.global_step = 1

        self.current_f1_score = float('-inf')
        self.best_f1_score = float('-inf')
        self.best_epoch = None

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
        pbar = tqdm(
            self.data_loader_train, desc="Training Loss: ?", leave=False)
        for step, data in enumerate(pbar, start=1):
            if self.cuda:
                with amp.autocast():
                    outputs, loss = self.train_forward(data)
            else:
                outputs, loss = self.train_forward(data)

            loss_normalized = loss / self.gradient_accumulation_steps
            if self.cuda:
                self.scaler.scale(loss_normalized).backward()
            else:
                loss_normalized.backward()

            if step % self.gradient_accumulation_steps == 0 \
                    or step >= len(self.data_loader_train):
                if self.cuda:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if self.global_step % self.steps_per_log == 0:
                self.train_log(outputs, loss)
                pbar.set_description(f"Training Loss: {loss.item():.3g}")

            self.global_step += 1
        self.lr_scheduler.step()

    def train_forward(self, data):
        data = recursive_to_device(data, self.device, non_blocking=True)

        images, labels = data
        images = self.data_loader_train.dataset.transforms(images)

        outputs = self.model(images)
        loss = self.model.loss(outputs, labels)
        return outputs, loss

    def train_log(self, outputs, loss):
        if self.writer is not None:
            self.writer.add_scalar('loss', loss, global_step=self.global_step)

    def eval(self):
        outputs_list = []
        labels_list = []

        pbar = tqdm(self.data_loader_eval, desc="Evaluation", leave=False)
        for step, data in enumerate(pbar, start=1):
            outputs, labels = self.eval_forward(data)

            outputs_list.append(outputs)
            labels_list.append(labels)
        outputs = torch.cat(outputs_list)
        labels = torch.cat(labels_list)

        self.eval_log(outputs, labels)

    def eval_forward(self, data):
        data = recursive_to_device(data, self.device, non_blocking=True)

        images, labels = data
        images = self.data_loader_eval.dataset.transforms(images)

        outputs = self.model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        outputs = torch.argmax(outputs, dim=-1)
        return outputs, labels

    def eval_log(self, outputs, labels):
        outputs = outputs.cpu().numpy()
        labels = labels.cpu().numpy()

        self.current_f1_score = f1_score(
            labels, outputs, average='macro', zero_division=0)

        if self.current_f1_score >= self.best_f1_score:
            self.best_f1_score = self.current_f1_score
            self.best_epoch = self.epoch

        if self.writer is not None:
            self.writer.add_scalar(
                'f1_score', self.current_f1_score, global_step=self.epoch)

        print(classification_report(
            labels,
            outputs,
            labels=range(len(self.data_loader_eval.dataset.labels)),
            target_names=self.data_loader_eval.dataset.labels,
            digits=5,
            zero_division=0))

    def save_model(self):
        if self.save_path is not None:
            self.model.save(
                self.save_path,
                epoch=self.epoch,
                f1_score=self.current_f1_score)
