#!/usr/bin/env python3

import math
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import plenoptic as po
import torch

from .models import MetamerMixture


def plot_representation_error(
    metamer: MetamerMixture, col_wrap: Optional[int] = None
) -> mpl.figure.Figure:
    """Plot representation of each input image, metamer, and representation error.

    The representations will all have the same ylim for each subplot, while the
    representation error's ylim will be different.

    Parameters
    ----------
    metamer :
        A MetamerMixture object that has completed synthesis using an instance
        of PortillaSimoncelliMinimalMixture
    col_wrap :
        How many columns to have in the image. If None, we use the number of
        input images (e.g., ``metamer.images.shape[0]``).

    Returns
    -------
    fig :
        Figure of images

    """
    # create dummy PS model, same args as minimal one, and run it on an image to create ps.representation
    # rep = ps_model(img, remove_stats=False) - ps_model(metamer.metamer, remove_stats=False)
    # ps_dummy.plot_representation(rep)
    ps_mixture = metamer.model
    ps_dummy = po.simul.PortillaSimoncelli(
        ps_mixture.image_shape,
        ps_mixture.n_scales,
        ps_mixture.n_orientations,
        ps_mixture.spatial_corr_width,
        ps_mixture.use_true_correlations,
    )
    # need to run the dummy model on an image to initialize some internal
    # variables
    ps_dummy(metamer.metamer)
    if col_wrap is None:
        col_wrap = metamer.image.shape[0]
    n_plots = metamer.image.shape[0] + 1
    n_rows = math.ceil(n_plots / col_wrap) + 1
    figsize = (12 * col_wrap, 5 * n_rows)
    fig, axes = plt.subplots(
        n_rows, col_wrap, sharex=True, sharey=True, figsize=figsize
    )
    titles = [
        f"Model representation of image {i}, weight={w.item():.3f}"
        for i, w in enumerate(metamer.model.im_weights)
    ]
    titles += ["Model representation of mixed texture metamer"]
    ylim = [0, 0]
    child_axes = []
    for t, ax, img in zip(
        titles, axes.flatten(), torch.cat([metamer.image, metamer.metamer])
    ):
        rep = ps_mixture(img.unsqueeze(0), remove_stats=False)
        _, rep_axes = ps_dummy.plot_representation(rep, ax=ax, ylim=False)
        ylim[0] = min([ylim[0], *[a.get_ylim()[0] for a in rep_axes]])
        ylim[1] = max([ylim[1], *[a.get_ylim()[1] for a in rep_axes]])
        child_axes.extend(rep_axes)
        ax.set_title(t, y=1.05)
    rep = ps_mixture(metamer.image, remove_stats=False) - ps_mixture(
        metamer.metamer, remove_stats=False
    )
    # because this is the error plot, it should be different ylim than
    # everyone else
    ps_dummy.plot_representation(rep, ax=axes[-1, 0], ylim=False)
    axes[-1, 0].set_title("Weighted Rep(inputs) -  Rep(mixed texture metamer)", y=1.05)
    for ax in axes.flatten():
        if not ax.get_title():
            ax.set_visible(False)
    for ax in child_axes:
        ax.set_ylim(ylim)
    return fig


def input_comparison(
    metamer: MetamerMixture,
    col_wrap: Optional[int] = None,
) -> mpl.figure.Figure:
    """Create plot showing the mixed images and resulting metamer.

    Parameters
    ----------
    metamer :
        A MetamerMixture object that has completed synthesis using an instance
        of PortillaSimoncelliMinimalMixture
    col_wrap :
        How many columns to have in the image. If None, we use the number of
        input images (e.g., ``metamer.images.shape[0]``).

    Returns
    -------
    fig :
        Figure of images

    """
    titles = [
        f"Input image {i}, weight={w.item():.3f}"
        for i, w in enumerate(metamer.model.im_weights)
    ]
    titles += ["Mixed texture metamer"]
    if col_wrap is None:
        col_wrap = metamer.image.shape[0]
    return po.imshow(
        torch.cat([metamer.image, metamer.metamer]),
        channel_idx=0,
        col_wrap=col_wrap,
        title=titles,
    )
