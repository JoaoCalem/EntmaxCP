from .TorchVisionDatasets import TorchVisionDatasets

import torchvision
import torchvision.transforms as transforms

class ImageNet(TorchVisionDatasets):
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