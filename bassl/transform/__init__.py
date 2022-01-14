import os

import torchvision.transforms as pth_transforms
from classy_vision.dataset.transforms import build_transform
from classy_vision.generic.registry_utils import import_all_modules


def get_transform(input_transforms_list):
    output_transforms = []
    for transform_config in input_transforms_list:
        transform = build_transform(transform_config)
        output_transforms.append(transform)
    return pth_transforms.Compose(output_transforms)


assert "PYTHONPATH" in os.environ
PROJ_ROOT = os.environ["PYTHONPATH"]

import_all_modules(os.path.join(PROJ_ROOT, "transform"), "transform")

__all__ = ["get_transform"]
