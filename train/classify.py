import torch
import torch.nn.functional as F
from tqdm import tqdm

from train.utils import recursive_to_device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def classify(data_loader, model):
    model = model.to(device)

    with torch.no_grad():
        outputs_list = []
        labels_list = []
        for data in tqdm(data_loader, leave=False):
            data = recursive_to_device(data, device, non_blocking=True)
            images, labels = data
            images = data_loader.dataset.transforms(images)

            outputs = model(images)
            if not isinstance(outputs, tuple):
                outputs = F.softmax(outputs, dim=-1)

            outputs_list.append(outputs)
            labels_list.append(labels)

        if isinstance(outputs_list[0], tuple):
            outputs_list = zip(*outputs_list)
            labels_list = zip(*labels_list)

            outputs = [torch.cat(outputs) for outputs in outputs_list]
            labels = [torch.cat(labels) for labels in labels_list]
        else:
            outputs = torch.cat(outputs_list)
            labels = torch.cat(labels_list)
        return outputs, labels
