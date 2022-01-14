import random
from typing import Any, Dict

import torchvision.transforms.functional as F
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image
from torchvision.transforms import RandomResizedCrop


@register_transform("VideoRandomResizedCrop")
class VideoRandomResizedCrop(ClassyTransform):
    def __init__(self, size, bottom_area=0.2):
        self.p = 1.0
        self.interpolation = Image.BICUBIC
        self.size = size
        self.bottom_area = bottom_area

    def __call__(self, imgmap):
        assert isinstance(imgmap, list)
        if random.random() < self.p:  # do RandomResizedCrop, consistent=True
            top, left, height, width = RandomResizedCrop.get_params(
                imgmap[0], scale=(self.bottom_area, 1.0), ratio=(3 / 4.0, 4 / 3.0)
            )
            return [
                F.resized_crop(
                    img=img,
                    top=top,
                    left=left,
                    height=height,
                    width=width,
                    size=(self.size, self.size),
                )
                for img in imgmap
            ]
        else:
            return [F.resize(img=img, size=[self.size, self.size]) for img in imgmap]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VideoRandomResizedCrop":
        size = config.get("size", 224)
        bottom_area = config.get("bottom_area", 0.14)
        return cls(size=size, bottom_area=bottom_area)
