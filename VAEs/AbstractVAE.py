import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractVAE(nn.Module, ABC):
    def __init__(self):
        super(AbstractVAE, self).__init__()

    @abstractmethod
    def encode(self, x):
        raise NotImplementedError

    @abstractmethod
    def reparameterize(self, enc_out):
        raise NotImplementedError

    @abstractmethod
    def decode(self, z):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError
