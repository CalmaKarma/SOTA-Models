import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class AbstractGAN(nn.Module, ABC):
    def __init__(self):
        super(AbstractGAN, self).__init__()
        # Instantiate generator and discriminator using abstract builder methods.
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    @abstractmethod
    def build_generator(self) -> nn.Module:
        """
        Constructs the generator network.
        Returns:
            An instance of torch.nn.Module representing the generator.
        """
        raise NotImplementedError

    @abstractmethod
    def build_discriminator(self) -> nn.Module:
        """
        Constructs the discriminator network.
        Returns:
            An instance of torch.nn.Module representing the discriminator.
        """
        raise NotImplementedError

    @abstractmethod
    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def discriminator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)
