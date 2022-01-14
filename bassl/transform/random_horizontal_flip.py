import random
from typing import Any, Dict

from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image


@register_transform("VideoRandomHFlip")
class VideoRandomHFlip(ClassyTransform):
    def __init__(self, consistent=True, command=None, seq_len=0):
        self.consistent = consistent
        if seq_len != 0:
            self.consistent = False
        if command == "left":
            self.threshold = 0
        elif command == "right":
            self.threshold = 1
        else:
            self.threshold = 0.5
        self.seq_len = seq_len

    def __call__(self, imgmap):
        assert isinstance(imgmap, list)
        if self.consistent:
            if random.random() < self.threshold:
                return [i.transpose(Image.FLIP_LEFT_RIGHT) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for idx, i in enumerate(imgmap):
                if idx % self.seq_len == 0:
                    th = random.random()
                if th < self.threshold:
                    result.append(i.transpose(Image.FLIP_LEFT_RIGHT))
                else:
                    result.append(i)
            assert len(result) == len(imgmap)
            return result

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VideoRandomHFlip":
        return cls()
