#!/usr/bin/env py3

import argparse
import json
import os
import os.path as op
import shutil
from datetime import datetime
from typing import Dict, Literal, Optional, Tuple

import git
import imageio
import numpy as np
import plenoptic as po
import torch
import yaml
from torch import Tensor

from . import display
from .models import MetamerMixture, PortillaSimoncelliMinimalMixture


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
    device: Literal["cpu", "gpu"],
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


def generate_texture(
    images_dict: Dict[str, float],
    metamer_synthesis_params: Dict = {
        "change_scale_criterion": None,
        "ctf_iters_to_check": 3,
    },
    opt_hyperparams: Dict = {"lr": 0.02, "amsgrad": True},
    model_params: Dict = {},
    img_init: str = "random",
    device: Literal["cpu", "gpu"] = "cpu",
) -> MetamerMixture:
    """Prepare images, model and generate a mixed texture!

    Parameters
    ----------
    images_dict :
        Dictionary where keys are paths to images to load and values are
        their relative weights.
    metamer_synthesis_params :
        Dictionary of parameters to pass to the MetamerMixture object when
        starting synthesis. See the docstring of Metamer's ``synthesis`` method
        for details.
    opt_hyperparams :
        Dictionary of hyper-parameters to pass to torch.optim.Adam on init.
    model_params :
        Dictionary of parameters to pass to PortillaSimoncelliMinimalMixture on
        init.
    img_init :
        How to initialize metamer synthesis, either "random" or a path. If
        random, we initialize with uniform noise from [m-.01, m+.01], where m
        is the overall mean of the input images.
    device :
        Which device to put tensors on.

    Returns
    -------
    metamer :
        The MetamerMixture object, with synthesis completed.

    """
    imgs, weights = prep_imgs(images_dict, device)
    # this is a good initial image for texture synthesis, though you can change
    # it if you'd like
    if img_init == "random":
        img_init = torch.rand_like(imgs[0].unsqueeze(0) * 0.01 + imgs.mean())
    else:
        img_init = po.load_images(img_init)
    ps = PortillaSimoncelliMinimalMixture(imgs.shape[-2:], weights, **model_params)
    ps.eval()
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
        **metamer_synthesis_params,
    )
    return met


# - add command-line function to run PSMinimal model on an image and save output?
#   - along with mixture?
# - add something to load in metamer, directly from output dir
def main(config_path: str, output_dir: str):
    """Create metamer, based on config, and save outputs.

    After synthesis finishes, we create a "timestamp" directory (DDMMYY_HHMMSS)
    within ``output_dir`` and save all of our outputs within that. Thus, if you
    have several jobs finishing at the *exact* same second, they will conflict
    and we thus raise an Exception.

    The timestamp directory contains:
    - ``versions.json``, a json file containing the git hash of this repo and
      the version of plenoptic used.
    - ``metamer.pt``, the actual metamer object.
    - the configuration file found at ``config_path`` (name unchanged)
    - ``metamer.png``, the texture metamer image, as an 8 bit
    - ``synthesis_status.svg``, plot showing the loss over time and the final metamer
    - ``input_comparison.svg``, plot the input images, with weights, and the final metamer
    - ``representation_error.svg``, plot the representation of each input image,
      the final metamer, and the representation error.

    Parameters
    ----------
    config_path :
        Path to a yml file containing the configuration options. See
        ``config.yml`` included in repo for an example.
    output_dir :
        Path to a containing directory, which we'll create the output directory
        in.

    """
    config = read_yml(config_path)
    met = generate_texture(**config)
    now = datetime.now().strftime("%d%m%y_%H%M%S")
    output_dir = op.join(output_dir, now)
    os.makedirs(output_dir)
    shutil.copy(config_path, output_dir)
    met.save(op.join(output_dir, "metamer.pt"))
    wishart_version = git.Repo(op.join(op.dirname(op.realpath(__file__)), '..')).head.object.hexsha
    with open(op.join(output_dir, "versions.json"), "w") as f:
        json.dump({"plenoptic": po.__version__, "wishart-textures": wishart_version}, f)
    # Convert metamer image to 8bit int and save
    metamer_img = np.clip(po.to_numpy(met.metamer.squeeze()), 0, 1)
    metamer_img = (metamer_img * np.iinfo(np.uint8).max).astype(np.uint8)
    imageio.imsave(op.join(output_dir, "metamer.png"), metamer_img)
    fig = po.synth.metamer.plot_synthesis_status(
        met, included_plots=["display_metamer", "plot_loss"]
    )[0]
    fig.savefig(op.join(output_dir, "synthesis_status.svg"),
                bbox_inches='tight')
    fig = display.input_comparison(met)
    fig.savefig(op.join(output_dir, "input_comparison.svg"))
    fig = display.plot_representation_error(met)
    fig.savefig(op.join(output_dir, "representation_error.svg"),
                bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""Generate some mixed textures!

    After synthesis finishes, we create a "timestamp" directory (DDMMYY_HHMMSS)
    within ``output_dir`` and save all of our outputs within that. Thus, if you
    have several jobs finishing at the *exact* same second, they will conflict
    and we thus raise an Exception.

    The timestamp directory contains:
    - versions.json, a json file containing the git hash of this repo and
      the version of plenoptic used.
    - metamer.pt, the actual metamer object.
    - the configuration file found at config_path (name unchanged)
    - metamer.png, the texture metamer image, as an 8 bit int
    - synthesis_status.svg, plot showing the loss over time and the final metamer
    - input_comparison.svg, plot the input images, with weights, and the final metamer
    - representation_error.svg, plot the representation of each input image,
      the final metamer, and the representation error.
    """,
    )
    parser.add_argument(
        "config_path",
        help="Path to a yml file containing the configuration options.",
    )
    parser.add_argument(
        "output_dir",
        help="Path to put the output directory in.",
    )
    args = vars(parser.parse_args())
    main(**args)
