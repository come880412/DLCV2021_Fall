import random
from typing import Callable

import pandas as pd
import torch
import torch.utils.data
import torchvision
import math


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, reweight=None, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()
        if reweight:
            theta = (random.randint(0, 90) / 180) * math.pi
            cos_comm = 0.7 + abs(math.cos(theta))
            cos_freq = 2.5 * (1 + abs(math.cos(theta)))
            cos_rare = 1/1.5 * (0.1 + abs(math.cos(theta)))
            print('reweight ratio f:%.3f, c:%.3f, r:%.3f' % (cos_freq, cos_comm, cos_rare))
            
            weights = 1.0 / label_to_count[df["label"]]
            self.weights = weights.to_list()

            reweights = self.weights.copy()

            for idx, i in enumerate(self.weights):
                if i > 0.1:
                    reweights[idx] = i * cos_rare
                elif i < 0.01:
                    reweights[idx] = i * cos_freq
                else:
                    reweights[idx] = i * cos_comm
            
            self.weights = torch.DoubleTensor(reweights)
        
        else:
            weights = 1.0 / label_to_count[df["label"]]
            weights = weights.to_list()

            self.weights = torch.DoubleTensor(weights)

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
