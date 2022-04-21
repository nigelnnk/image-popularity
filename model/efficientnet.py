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


class EfficientNet(BaseModel):
    def __init__(
            self,
            efficientnet_model_name,
            num_outputs,
            dropout_rate=0.1,
            efficientnet_pretrained=False):
        # Initialise BaseModel with model args and kwargs
        # Used saving and loading model
        # Set efficientnet_pretrained to False
        super(EfficientNet, self).__init__(
            efficientnet_model_name=efficientnet_model_name,
            num_outputs=num_outputs,
            dropout_rate=dropout_rate,
            efficientnet_pretrained=False)

        self.efficientnet_model_name = efficientnet_model_name
        self.dropout_rate = dropout_rate
        self.num_outputs = num_outputs

        efficientnet = EFFICIENTNET_MODELS[self.efficientnet_model_name]
        self.efficientnet = efficientnet(pretrained=efficientnet_pretrained)

        for stage in range(7):
            self.efficientnet.features[stage].requires_grad_(False)

        classifier_input_size = self.efficientnet.classifier[-1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate, inplace=True),
            nn.Linear(classifier_input_size, self.num_outputs),
        )

    def init_parameters(self):
        pass

    def reset_parameters(self):
        pass

    def forward(self, images):
        outputs = self.efficientnet(images)
        return outputs

    def loss(self, outputs, labels):
        loss = F.cross_entropy(outputs, labels)
        return loss
