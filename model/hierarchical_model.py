import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.base_model import BaseModel


EFFICIENTNET_MODELS = {
    'efficientnet_b0': models.efficientnet_b0,
    'efficientnet_b1': models.efficientnet_b1,
    'efficientnet_b2': models.efficientnet_b2,
    'efficientnet_b3': models.efficientnet_b3,
    'efficientnet_b4': models.efficientnet_b4,
    'efficientnet_b5': models.efficientnet_b5,
    'efficientnet_b6': models.efficientnet_b6,
    'efficientnet_b7': models.efficientnet_b7,
}


class EfficientNetBackbone(nn.Module):
    def __init__(self, efficientnet_model_name, num_stages, pretrained=False):
        super(EfficientNetBackbone, self).__init__()

        efficientnet = EFFICIENTNET_MODELS[efficientnet_model_name]
        efficientnet = efficientnet(pretrained=pretrained)
        efficientnet_stages = efficientnet.features

        self.stages = nn.Sequential(*efficientnet_stages[:num_stages])

    def forward(self, x):
        return self.stages(x)


class EfficientNetHead(nn.Module):
    def __init__(self, efficientnet_model_name, num_stages, pretrained=False):
        super(EfficientNetHead, self).__init__()

        efficientnet = EFFICIENTNET_MODELS[efficientnet_model_name]
        efficientnet = efficientnet(pretrained=pretrained)
        efficientnet_stages = efficientnet.features

        self.stages = nn.Sequential(*efficientnet_stages[-num_stages:])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stages(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class HierarchicalEfficientNet(BaseModel):
    """Hierarchical EfficientNet based on HD-CNN.

    https://arxiv.org/abs/1410.0736
    """
    def __init__(
            self,
            efficientnet_model_name,
            num_coarse_outputs,
            num_fine_outputs,
            dropout_rate=0.1,
            efficientnet_pretrained=False):
        # Initialise BaseModel with model args and kwargs
        # Used saving and loading model
        # Set efficientnet_pretrained to False
        super(HierarchicalEfficientNet, self).__init__(
            efficientnet_model_name=efficientnet_model_name,
            num_coarse_outputs=num_coarse_outputs,
            num_fine_outputs=num_fine_outputs,
            dropout_rate=dropout_rate,
            efficientnet_pretrained=False)

        self.efficientnet_model_name = efficientnet_model_name
        self.num_coarse_outputs = num_coarse_outputs
        self.dropout_rate = dropout_rate
        self.num_fine_outputs = num_fine_outputs

        self._mode = 'hierarchical'
        self._fine_category = None

        self.backbone = EfficientNetBackbone(
            efficientnet_model_name, 7, pretrained=efficientnet_pretrained)
        self.backbone.requires_grad_(False)

        head = EfficientNetHead(
            efficientnet_model_name, 2, pretrained=efficientnet_pretrained)
        head_output_channels = head.stages[-1][0].out_channels

        self.coarse_head = nn.Sequential(
            copy.deepcopy(head),
            nn.Dropout(p=self.dropout_rate, inplace=True),
            nn.Linear(head_output_channels, num_coarse_outputs),
        )
        self.fine_heads = nn.ModuleList([
            nn.Sequential(
                copy.deepcopy(head),
                nn.Dropout(p=self.dropout_rate, inplace=True),
                nn.Linear(head_output_channels, num_fine_outputs),
            )
            for _ in range(num_coarse_outputs)])

    def init_parameters(self):
        pass

    def reset_parameters(self):
        pass

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        self._mode = new_mode

    @property
    def fine_category(self):
        return self._fine_category

    @fine_category.setter
    def fine_category(self, new_fine_category):
        self._fine_category = new_fine_category

    @staticmethod
    def probabilistic_averaging_layer(coarse_outputs, fine_outputs):
        coarse_outputs = F.softmax(coarse_outputs, dim=1)
        fine_outputs = torch.stack(
            [F.softmax(outputs, dim=1) for outputs in fine_outputs],
            dim=1)
        coarse_outputs = torch.unsqueeze(coarse_outputs, -1)
        outputs = torch.sum(coarse_outputs * fine_outputs, dim=1)
        return outputs

    def forward(self, images):
        shared_features = self.backbone(images)

        if self.mode == 'coarse':
            coarse_outputs = self.coarse_head(shared_features)
            return coarse_outputs
        elif self.mode == 'fine':
            fine_head = self.fine_heads[self.fine_category]
            fine_outputs = fine_head(shared_features)
            return fine_outputs
        else:
            coarse_outputs = self.coarse_head(shared_features)
            fine_outputs = [
                fine_head(shared_features) for fine_head in self.fine_heads]
            outputs = self.probabilistic_averaging_layer(
                coarse_outputs, fine_outputs)
            return outputs, coarse_outputs

    def loss(self, outputs, labels):
        if self.mode == 'coarse' or self.mode == 'fine':
            loss = F.cross_entropy(outputs, labels)
        else:
            outputs, coarse_outputs = outputs
            labels, coarse_labels = labels
            loss = F.nll_loss(torch.log(outputs), labels)
            coarse_loss = F.nll_loss(torch.log(coarse_outputs), coarse_labels)
            loss = loss + coarse_loss
        return loss
