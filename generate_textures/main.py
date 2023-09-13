#!/usr/bin/env py3

import argparse
import json
import os
import os.path as op
import shutil
from datetime import datetime
from typing import Dict, Literal

import git
import imageio
import numpy as np
import plenoptic as po
import torch
from torch import Tensor

from . import display, utils
from .models import MetamerMixture, PortillaSimoncelliMinimalMixture


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
    imgs, weights = utils.prep_imgs(images_dict, device)
    # this is a good initial image for texture synthesis, though you can change
    # it if you'd like
    if img_init == "random":
        img_init = torch.rand_like(imgs[0].unsqueeze(0)) * 0.01 + imgs.mean()
    else:
        img_init = po.load_images(img_init)
    ps = PortillaSimoncelliMinimalMixture(imgs.shape[-2:], weights, **model_params).to(device)
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
    - ``metamer.pt``, the actual metamer object. To examine it, use the
      ``generate_textures.models.load_metamer_mixture`` function. See
      plenoptic's documentation for more details on how to interact with
      metamer objects.
    - ``metamer_representation.pt``, torch.Tensor of shape (1, 1, s) (where
      s~=700) giving the representation of the synthesized metamer. Load in
      using ``torch.load``.
    - ``mixed_input_image_representation.pt``, torch.Tensor of shape (1, 1, s)
      (where s~=700) giving the weighted representation of the input images. If
      synthesis converged and went perfectly, this should be identical to
      ``metamer_representation.pt``.  Load in using ``torch.load``.
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
    config = utils.read_yml(config_path)
    met = generate_texture(**config)
    # move everything over to the cpu for plotting and saving
    met.to('cpu')
    now = datetime.now().strftime("%d%m%y_%H%M%S")
    output_dir = op.join(output_dir, now)
    os.makedirs(output_dir)
    shutil.copy(config_path, output_dir)
    met.save(op.join(output_dir, "metamer.pt"))
    met_rep = met.model(met.metamer)
    torch.save(met_rep, op.join(output_dir, "metamer_representation.pt"))
    mixed_rep = met.model(met.image)
    torch.save(mixed_rep, op.join(output_dir, "mixed_input_image_representation.pt"))
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
    - metamer.pt, the actual metamer object. To examine it, use the
      generate_textures.models.load_metamer_mixture function. See plenoptic's
      documentation for more details on how to interact with metamer objects.
    - metamer_representation.pt, torch.Tensor of shape (1, 1, s) (where
      s~=700) giving the representation of the synthesized metamer. Load in
      using torch.load
    - mixed_input_image_representation.pt, torch.Tensor of shape (1, 1, s)
      (where s~=700) giving the weighted representation of the input images. If
      synthesis converged and went perfectly, this should be identical to
      ``metamer_representation.pt``. Load in using torch.load
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
