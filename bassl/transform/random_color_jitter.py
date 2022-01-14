import numbers
import random
from typing import Any, Dict

import torchvision
import torchvision.transforms.functional as F
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("VideoRandomColorJitter")
class VideoRandomColorJitter(ClassyTransform):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(
        self,
        brightness=0,
        contrast=0,
        saturation=0,
        hue=0,
        consistent=True,
        p=1.0,
        seq_len=0,
    ):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.consistent = consistent
        self.threshold = p
        self.seq_len = seq_len

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with lenght 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(
                torchvision.transforms.Lambda(
                    lambda img: F.adjust_brightness(img, brightness_factor)
                )
            )

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(
                torchvision.transforms.Lambda(
                    lambda img: F.adjust_contrast(img, contrast_factor)
                )
            )

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                torchvision.transforms.Lambda(
                    lambda img: F.adjust_saturation(img, saturation_factor)
                )
            )

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(
                torchvision.transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor))
            )

        random.shuffle(transforms)
        transform = torchvision.transforms.Compose(transforms)

        return transform

    def __call__(self, imgmap):
        assert isinstance(imgmap, list)
        if random.random() < self.threshold:  # do ColorJitter
            if self.consistent:
                transform = self.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue
                )
                return [transform(i) for i in imgmap]
            else:
                if self.seq_len == 0:
                    return [
                        self.get_params(
                            self.brightness, self.contrast, self.saturation, self.hue
                        )(img)
                        for img in imgmap
                    ]
                else:
                    result = []
                    for idx, img in enumerate(imgmap):
                        if idx % self.seq_len == 0:
                            transform = self.get_params(
                                self.brightness,
                                self.contrast,
                                self.saturation,
                                self.hue,
                            )
                        result.append(transform(img))
                    return result

                # result = []
                # for img in imgmap:
                #     transform = self.get_params(self.brightness, self.contrast,
                #                                 self.saturation, self.hue)
                #     result.append(transform(img))
                # return result
        else:  # don't do ColorJitter, do nothing
            return imgmap

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "brightness={0}".format(self.brightness)
        format_string += ", contrast={0}".format(self.contrast)
        format_string += ", saturation={0}".format(self.saturation)
        format_string += ", hue={0})".format(self.hue)
        return format_string

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VideoRandomColorJitter":
        brightness = config.get("brightness", 0.8)
        contrast = config.get("contrast", 0.8)
        saturation = config.get("saturation", 0.8)
        hue = config.get("hue", 0.2)
        p = config.get("p", 0.8)
        return cls(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=p,
        )
