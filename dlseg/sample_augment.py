import torch as t
import torch.nn as nn
import torchvision.transforms as tvf
from random import random, choice

class RandomIntensity(nn.Module):
    def __init__(self, amount):
        super(RandomIntensity, self).__init__()
        self.amount = amount

    def forward(self, x):
        if random() > 0.5:
            x = x * (1 + random() * self.amount * 2 - self.amount)
        if random() > 0.5:
            x = x + (random() * self.amount * 2 - self.amount)
        return x

class RandomNoise(nn.Module):
    def __init__(self, amount):
        super(RandomNoise, self).__init__()
        self.amount = amount

    def forward(self, x):
        if random() > 0.5:
            x = x + t.randn_like(x) * random() * self.amount
        return x

class RandomNormalize(nn.Module):
    def __init__(self):
        super(RandomNormalize, self).__init__()

    def forward(self, x):
        if random() > 0.5:
            x = (x - x.mean()) / (x.std() + 1e-8)
        return x

class Sampler_Augmenter():
    def __init__(
        self,
        tile_size,
        random_rotate = 180,
        random_intensity = 0.25,
        random_noise = 0.33,
        random_normalize = True,
    ):
        self.sampler = self._get_sampler(random_rotate, tile_size)
        self.augmenter = self._get_augmenter(random_normalize, random_intensity, random_noise)

    def _get_sampler(self, random_rotate, tile_size):
        return nn.Sequential(
            *(
                [
                    #tvf.RandomResizedCrop(int(tile_size * 1.5), scale=(0.5, 1.0), ratio=(0.75, 1.33)),
                    tvf.RandomCrop(int(tile_size * 1.5)),
                    tvf.RandomRotation(random_rotate),
                    tvf.CenterCrop(tile_size),
                ]
                if random_rotate > 0
                else [tvf.RandomCrop(tile_size)]
            )
        )

    def _get_augmenter(self, random_normalize, random_intensity, random_noise):
        modules = []
        if random_normalize:
            modules += [RandomNormalize()]
        if random_intensity > 0:
            modules += [RandomIntensity(random_intensity)]
        if random_noise > 0:
            modules += [RandomNoise(random_noise)]
        return nn.Sequential(*modules)

    def get_batch(self,train_tsrs,batch_size):
        batch = t.cat(
            [self.sampler(choice(train_tsrs)) for _ in range(batch_size)],
            dim=0,
        )

        for i in range(batch_size):
            batch[i, 0:1] = self.augmenter(batch[i, 0:1])
        return batch

