from collections import abc

import torch


def recursive_to_device(data, device, non_blocking=False):
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, abc.Mapping):
        return {
            key: recursive_to_device(value, device, non_blocking=non_blocking)
            for key, value in data.items()}
    elif isinstance(data, abc.Sequence) and not isinstance(data, str):
        return [
            recursive_to_device(item, device, non_blocking=non_blocking)
            for item in data]
    else:
        return data
