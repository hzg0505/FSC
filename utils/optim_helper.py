import torch
from torch.optim.lr_scheduler import ExponentialLR, StepLR
# 3. 优化器
def get_optimizer(parameters, config):
    if config['type'] == 'Adam':
        return torch.optim.Adam(parameters, **config['kwargs'])
    elif config['type'] == "SGD":
        return torch.optim.SGD(parameters, **config['kwargs'])
    elif config['type'] == "AdamW":
        return torch.optim.AdamW(parameters, **config['kwargs'])
    else:
        raise NotImplementedError


def get_scheduler(optimizer, config):
    if config["type"] == "StepLR":
        return StepLR(optimizer, **config["kwargs"])
    elif config["type"] == "ExponentialLR":
        return ExponentialLR(optimizer, **config["kwargs"])
    else:
        raise NotImplementedError