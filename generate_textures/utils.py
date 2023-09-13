#!/usr/bin/env python3

import yaml
from typing import Dict, Literal, Tuple
import torch
import plenoptic as po
from torch import Tensor

def read_yml(config_path: str) -> dict:
    """Read config from path, parsing Nones and scientific notation."""
    with open(config_path) as f:
        kwargs = yaml.safe_load(f.read())
    for k, v in kwargs.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if vv == "None":
                    kwargs[k][kk] = None
                elif isinstance(vv, str):
                    try:
                        kwargs[k][kk] = float(vv)
                    except ValueError:
                        pass
        else:
            if v == "None":
                kwargs[k] = None
            elif isinstance(v, str):
                try:
                    kwargs[k] = float(v)
                except ValueError:
                    pass
    return kwargs


def prep_imgs(
    imgs_dict: Dict[str, float],
    device: Literal["cpu", "gpu"] = 'cpu',
) -> Tuple[Tensor, Tensor]:
    """Prepare images.

    All images are grayscale and rescaled to lie in the range [0, 1].

    If any of the weights in imgs_dict are 0, we skip that image.

    If any weight is negative or if their sum is not 1, this raises a
    ValueError.

    Parameters
    ----------
    imgs_dict :
        Dictionary where keys are paths to images to load and values are
        their relative weights.
    device :
        Which device to put tensors on.

    Returns
    -------
    images :
        4d tensor of images of shape (k, 1, h, w).
    weights :
        1d tensor of weights of shape (k,)

    """
    imgs = []
    weights = []
    for k, v in imgs_dict.items():
        if v == 0:
            continue
        imgs.append(k)
        weights.append(v)
    imgs = po.load_images(imgs).to(device)
    weights = torch.Tensor(weights).to(device)
    if (weights < 0).any():
        raise ValueError("Negative weight!")
    if weights.sum() != 1:
        raise ValueError("Weights must sum to 1!")
    return imgs, weights
