import torch
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from train.base_trainer import BaseTrainer
from train.utils import recursive_to_device


class AlexNet_Trainer(BaseTrainer):
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
            save_path=None,
            target="subreddits"):
        super().__init__(
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
        self.current_f1_score = float('-inf')
        self.best_f1_score = float('-inf')
        self.best_epoch = None
        self.target = target

    def train_forward(self, data):
        data = recursive_to_device(data, self.device, non_blocking=True)

        images, subreddits, bins = data
        images = self.data_loader_train.dataset.transforms(images)
        targets = subreddits
        if self.target in ["percentile", "log"]:
            targets = bins
        elif self.target == "mix":
            targets = subreddits*3 + bins

        outputs = self.model(images)
        loss = self.model.loss(outputs, targets)
        return outputs, loss

    def train_log(self, outputs, loss):
        if self.writer is not None:
            self.writer.add_scalar('loss', loss, global_step=self.global_step)

    def eval_forward(self, data):
        data = recursive_to_device(data, self.device, non_blocking=True)

        images, subreddits, bins = data
        images = self.data_loader_eval.dataset.transforms(images)

        outputs = torch.argmax(self.model(images), dim=-1)
        labels = subreddits
        if self.target in ["percentile", "log"]:
            labels = bins
        elif self.target == "mix":
            labels = subreddits*3 + bins
        
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

        if self.target == "percentile":
            target_names = [str(x) for x in self.data_loader_eval.dataset.percentile_bins]
        elif self.target == "log":
            target_names = ["10^1 (bad)", "10^2 (avg)", "10^3  (gd)"]
        elif self.target == "mix":
            mr = self.data_loader_eval.dataset.subreddits
            percent = [str(x) for x in self.data_loader_eval.dataset.percentile_bins]
            target_names = [x+" "+y for x in mr for y in percent]
        else:
            target_names = self.data_loader_eval.dataset.subreddits

        print(classification_report(
            labels,
            outputs,
            target_names=target_names,
            digits=5,
            zero_division=0))
        if self.target == "multireddit":
            y_true = labels
            print(confusion_matrix(y_true, outputs))

    def save_model(self):
        if self.save_path is not None and \
                self.current_f1_score >= self.best_f1_score:
            self.model.save(
                self.save_path,
                epoch=self.best_epoch,
                f1_score=self.best_f1_score)