from typing import Any, Dict

import torchvision.transforms as TF
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image


@register_transform("VideoResizedCenterCrop")
class VideoResizedCenterCrop(ClassyTransform):
    def __init__(self, image_size, crop_size):
        self.tfm = TF.Compose(
            [
                TF.Resize(size=image_size, interpolation=Image.BICUBIC),
                TF.CenterCrop(crop_size),
            ]
        )

    def __call__(self, imgmap):
        assert isinstance(imgmap, list)
        return [self.tfm(img) for img in imgmap]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VideoResizedCenterCrop":
        image_size = config.get("image_size", 256)
        crop_size = config.get("crop_size", 224)
        return cls(image_size=image_size, crop_size=crop_size)
