#!/usr/bin/env py3

import torch
from .models import PortillaSimoncelliMinimalMixture, MetamerMixture
import plenoptic as po
from typing import Dict, Literal, Optional
import yaml


def read_yml(config_path: str):
    """Read config from path and parse Nones. """
    with open(config_path) as f:
        kwargs = yaml.safe_load(f.read())
    for k, v in kwargs.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if vv == 'None':
                    kwargs[k][kk] = None
        if v == 'None':
            kwargs[k] = None
    return kwargs
            

def prep_imgs(imgs_dict: Dict[str, float], device: Literal['cpu', 'gpu']):
    """Prepare images.

    All images are grayscale and rescaled to lie in the range [0, 1].

    If any weight is negative or if their sum is not 1, this raises a
    ValueError.

    Parameters
    ----------
    imgs_dict :
        Dictionary where keys are paths to images to load and values are
        their relative weights.
    device :
        Which device to put tensors on.

    """
    imgs = []
    weights = []
    for k, v in imgs_dict.items():
        imgs.append(k)
        weights.append(v)
    imgs = po.load_images(imgs).to(device)
    weights = torch.Tensor(weights).to(device)
    if (weights < 0).any():
        raise ValueError("Negative weight!")
    if weights.sum() != 1:
        raise ValueError("Weights must sum to 1!")
    return imgs, weights


def generate_texture(
    images_dict: Dict[str, float],
    synth_iter: int = 1000,
    change_scale_criterion: Optional[float]=None,
    ctf_iters_to_check: int = 3,
    opt_hyperparams: Dict = {"lr": 0.02, "amsgrad": True},
    model_params: Dict = {},
    img_init: str = "random",
    device: Literal['cpu', 'gpu'] = "cpu",
):
    """Prepare images, model and generate a mixed texture!

    Parameters
    ----------
    images_dict :
        Dictionary where keys are paths to images to load and values are
        their relative weights.
    synth_iter :
        The maximum number of iterations to run before we end synthesis.
    change_scale_criterion
        Scale-specific analogue of ``change_scale_criterion``: we consider
        a given scale finished (and move onto the next) if the loss has
        changed less than this in the past ``ctf_iters_to_check``
        iterations. If ``None``, we'll change scales as soon as we've spent
        ``ctf_iters_to_check`` on a given scale.
    ctf_iters_to_check
        Scale-specific analogue of ``stop_iters_to_check``: how many
        iterations back in order to check in order to see if we should
        switch scales.
    opt_hyperparams :
        Dictionary of hyper-parameters to pass to torch.optim.Adam on init.
    model_params :
        Dictionary of parameters to pass to PortillaSimoncelliMinimalMixture on
        init.
    img_init :
        How to initialize metamer synthesis, either "random" or a path. If
        random, we initialize with uniform noise from [m-.01, m+.01], where m
        is the overall mean as the input images.
    device :
        Which device to put tensors on.

    """
    imgs, weights = prep_imgs(images_dict, device)
    # this is a good initial image for texture synthesis, though you can change
    # it if you'd like
    if img_init == "random":
        img_init = torch.rand_like(imgs[0].unsqueeze(0) * 0.01 + imgs.mean())
    else:
        img_init = po.load_images(img_init)
    ps = PortillaSimoncelliMinimalMixture(imgs.shape[-2:], weights, **model_params)
    met = MetamerMixture(
        imgs,
        ps,
        po.tools.optim.l2_norm,
        initial_image=img_init.to(device),
        coarse_to_fine="together",
    )
    opt = torch.optim.Adam([met.metamer], **opt_hyperparams)
    met.synthesize(
        optimizer=opt,
        max_iter=synth_iter,
        change_scale_criterion=change_scale_criterion,
        ctf_iters_to_check=ctf_iters_to_check,
    )
    return met

# ADD:
# - create output plots:
#   - imshow with input images (relative weights in title) and metamer
#   - representation of each input image, then representation error (maybe in separate)?
#   - synthesis status with just metamer and loss and pixel vals?
# - create output dir: copy config file, add readme
#   with git hash of this and plenoptic version
# - in that dir, save: met, output plots, metamer image itself
def main(config_path: str, output_dir: str):
    """
    """
    config = read_yml(config_path)
    met = generate_texture(**config)
