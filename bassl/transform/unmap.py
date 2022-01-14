from typing import Any, Dict

import numpy as np
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image

try:
    import accimage
except ImportError:
    accimage = None


# See /pytorch/vision/torchvision/transforms/functional.py
def _is_numpy(img):
    return isinstance(img, np.ndarray)


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


@register_transform("VideoUnmap")
class VideoUnmap(ClassyTransform):
    def __init__(self, mean=None, std=None, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

        assert self.mean is not None
        assert self.std is not None

    def __unmap__(self, img):
        for i in range(3):
            img[i] = (img[i] * self.std[i]) + self.mean[i]
        return img

    def __call__(self, imgmap):
        assert isinstance(imgmap, list)
        return [self.__unmap__(img) for img in imgmap]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VideoUnmap":
        mean = config.get("mean", [0.485, 0.456, 0.406])
        std = config.get("std", [0.229, 0.224, 0.225])
        return cls(mean=mean, std=std)
