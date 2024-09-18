import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math

from transformers import AutoImageProcessor

from abc import ABC, abstractmethod

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Datasets(ABC):
    def __init__(
            self,
            valid_ratio: float,
            batch_size: int,
            calibration_samples: int = 3000,
            transform: str = 'norm'
            ):

        train_dataset = self._get_dataset(transform, train=True)
        self._train_splits(train_dataset,
                           calibration_samples,valid_ratio, batch_size)

        test_dataset = self._get_dataset(transform, train=False)
        self._test = DataLoader(test_dataset, batch_size=batch_size)
        
        if transform=='vit':
            self.vit_processor = AutoImageProcessor.from_pretrained(
                            "google/vit-base-patch16-224",use_fast=True)
    
    @property
    def train(self):
        return self._train
    
    @property
    def dev(self):
        return self._dev
    
    @property
    def cal(self):
        return self._cal
    
    @property
    def test(self):
        return self._test
    
    @abstractmethod
    def _dataset_class(self):
        pass

    def _dataset(self, transform):
        data_class, normalize = self._dataset_class()
        
        if transform == 'norm':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                normalize])
        elif transform == 'vit':
            transform = lambda x: self.vit_processor(x.convert('RGB'))['pixel_values'][0]
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        
        return data_class, transform
    
    def _get_dataset(self, norm, train=True):
        data_class, transform = self._dataset(norm)
        return data_class(
            root="data",
            train=train,
            download=True,
            transform=transform
        )

    def _train_splits(self, train_dataset,
                      calibration_samples, valid_ratio, batch_size):
        
        gen = torch.Generator()
        gen.manual_seed(0)
        
        train_dataset, cal_dataset = torch.utils.data.dataset.random_split(
            train_dataset, 
            [len(train_dataset)-calibration_samples, calibration_samples],
            generator=gen
        )

        nb_train = int(math.ceil((1.0 - valid_ratio) * len(train_dataset)))
        nb_valid = int(math.floor((valid_ratio * len(train_dataset))))
        train_dataset, dev_dataset = torch.utils.data.dataset.random_split(
            train_dataset, [nb_train, nb_valid], generator=gen
        )
        
        self._train = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self._dev = DataLoader(dev_dataset, batch_size=batch_size)
        self._cal = DataLoader(cal_dataset, batch_size=batch_size)