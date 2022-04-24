import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from model.base_model import BaseModel


class AlexNet(BaseModel):
    def __init__(
            self,
            output_length,
            use_pretrained=False):
        # Initialise BaseModel with model args and kwargs
        # Used saving and loading model
        super().__init__(
            output_length=output_length)

        self.output_length = output_length
        self.pretrained = use_pretrained

        self.init_parameters()

    def init_parameters(self):
        if self.pretrained:
            self.model_ft = models.alexnet(pretrained=self.pretrained)
            for stage in range(10):
                self.model_ft.features[stage].requires_grad_(False)
        else:
            self.model_ft = models.alexnet()

        num_ftrs = self.model_ft.classifier[6].in_features
        self.model_ft.classifier[6] = nn.Linear(num_ftrs, self.output_length)

    def reset_parameters(self):
        self.init_parameters()

    def forward(self, features):
        return self.model_ft(features)

    def loss(self, outputs, labels):
        loss = F.cross_entropy(outputs, labels)
        return loss
