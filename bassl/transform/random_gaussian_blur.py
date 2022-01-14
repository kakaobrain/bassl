import random
from typing import Any, Dict

from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import ImageFilter


@register_transform("VideoRandomGaussianBlur")
class VideoRandomGaussianBlur(ClassyTransform):
    def __init__(self, radius_min=0.1, radius_max=2.0, p=0.5):
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.p = p

    def __call__(self, imgmap):
        assert isinstance(imgmap, list)
        if random.random() < self.p:
            result = []
            for _, img in enumerate(imgmap):
                _radius = random.uniform(self.radius_min, self.radius_max)
                result.append(img.filter(ImageFilter.GaussianBlur(radius=_radius)))
            return result
        else:
            return imgmap

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VideoRandomGaussianBlur":
        radius_min = config.get("radius_min", 0.1)
        radius_max = config.get("radius_max", 2.0)
        p = config.get("p", 0.5)
        return cls(radius_min=radius_min, radius_max=radius_max, p=p)
