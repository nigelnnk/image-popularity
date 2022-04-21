import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_model import BaseModel


class DummyModel(BaseModel):
    def __init__(
            self,
            input_channels,
            hidden_channels,
            output_channels):
        # Initialise BaseModel with model args and kwargs
        # Used saving and loading model
        super(DummyModel, self).__init__(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels)

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.activation = nn.ReLU()

        self.conv_1 = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(self.hidden_channels)

        self.conv_2 = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(self.hidden_channels)

        self.conv_3 = nn.Conv2d(
            self.hidden_channels,
            self.output_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True)

        self.init_parameters()

    def init_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.conv_1.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.conv_2.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.conv_3.weight, nonlinearity='linear')
            nn.init.constant_(self.conv_3.bias, 0)

    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.batch_norm_1.reset_parameters()

        self.conv_2.reset_parameters()
        self.batch_norm_2.reset_parameters()

        self.conv_3.reset_parameters()

        self.init_parameters()

    def forward(self, features):
        features = self.conv_1(features)
        features = self.batch_norm_1(features)
        features = self.activation(features)

        features = self.conv_2(features)
        features = self.batch_norm_2(features)
        features = self.activation(features)

        features = self.conv_3(features)
        features = torch.mean(features, dim=(2, 3))
        return features

    def loss(self, outputs, labels):
        loss = F.cross_entropy(outputs, labels)
        return loss
