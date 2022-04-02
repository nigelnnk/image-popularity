from abc import ABC, abstractmethod
import json
from pathlib import Path

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        """Initialise BaseModel with model args and kwargs.

        Used saving and loading model.
        """
        super(BaseModel, self).__init__()
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def init_parameters(self):
        """Initialise model parameters."""
        pass

    @abstractmethod
    def reset_parameters(self):
        """Reset model parameters."""
        pass

    @property
    def parameter_dicts(self):
        """Parameter dicts to pass to optimizer.

        Can be overridden for custom per-parameter options.
        """
        parameter_dicts = [{'params': self.parameters()}]
        return parameter_dicts

    def save(self, model_path, **config_update):
        """Save model args, kwargs, and state dict."""
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        config_path = model_path / 'config.json'
        state_dict_path = model_path / 'state_dict.pt'

        config = {
            'name': type(self).__name__,
            'args': self.args,
            'kwargs': self.kwargs,
        }
        config.update(config_update)
        with config_path.open('w') as file:
            json.dump(config, file, indent=4)

        torch.save(self.state_dict(), state_dict_path)

    @classmethod
    def load(cls, model_path, map_location=None, **config_update):
        """Load model from args, kwargs, and state dict."""
        model_path = Path(model_path)
        config_path = model_path / 'config.json'
        state_dict_path = model_path / 'state_dict.pt'

        with config_path.open() as file:
            config = json.load(file)
        config.update(config_update)

        model = cls(*config['args'], **config['kwargs'])
        model.load_state_dict(torch.load(
            state_dict_path, map_location=map_location))
        return model
