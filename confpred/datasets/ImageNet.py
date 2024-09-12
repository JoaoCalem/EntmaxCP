from confpred.datasets import Datasets

import torchvision
import torchvision.transforms as transforms

class ImageNet(Datasets):
    def __init__(
            self,
            valid_ratio: float,
            batch_size: int,
            calibration_samples: int = 3000,
            norm: bool = True
            ):
        super().__init__(valid_ratio, batch_size, calibration_samples, norm)
    
    def _dataset_class(self):
        data_class = torchvision.datasets.ImageNet
        normalize = transforms.Normalize(0.5, 0.5, 0.5)
        return data_class, normalize
    
    def _get_dataset(self, norm, train=True):
        split = 'val'
        if train:
            split='train'
        data_class, transform = self._dataset(norm)
        return data_class(
            root="data/imagenet",
            split=split,
            transform=transform
        )
        
    def _dataset(self, norm):
        data_class, normalize = self._dataset_class()
        
        if norm:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Resize([256,256]),
                normalize,])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        
        return data_class, transform