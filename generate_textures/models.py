#!/usr/bin/env python3

from collections import OrderedDict

import numpy as np
from glob import glob
import os.path as op
import plenoptic as po
import torch

from . import utils


class PortillaSimoncelliMixture(po.simul.PortillaSimoncelli):
    r"""Extend the PortillaSimoncelli model to mix arbitrary combinations of images

    This model can accept multiple images, indexed on the batch
    dimension (remember, inputs have shape (batch, channel, height, width)).
    The model is initialized with a weight vector, which tells us how to weight
    the outputs of these images, so as to produce a mixture image. It can also
    accept a single image, in which case it behaves as normal.

    This is based off the ``PortillaSimoncelliMixture`` classes found at the
    end of the PS texture metamer notebook, but here we allow for an arbitrary
    number of images, with variable weighting

    Parameters
    ----------
    im_shape: int
        the size of the images being processed by the model
    im_weights: torch.Tensor
        How to weight the mixture of images when given multiple images
    n_scales: int
        number of scales of the steerable pyramid
    n_orientations: int
        number of orientations of the steerable pyramid
    spatial_corr_width: int
        width of the spatial correlation window

    """
    def __init__(
        self,
        im_shape,
        im_weights,
        n_scales=4,
        n_orientations=4,
        spatial_corr_width=7,
    ):
        super().__init__(
            im_shape,
            n_scales=n_scales,
            n_orientations=n_orientations,
            spatial_corr_width=spatial_corr_width,
        )
        self.im_weights = im_weights

    def forward(self, images, scales=None):
        r"""Average Texture Statistics representations of two image

        Parameters
        ----------
        images : torch.Tensor
            A 4d tensor containing one or two images to analyze, with shape (i,
            channel, height, width), i in {1,2}.
        scales : list, optional
            Which scales to include in the returned representation. If an empty
            list (the default), we include all scales. Otherwise, can contain
            subset of values present in this model's ``scales`` attribute.

        Returns
        -------
        representation_tensor: torch.Tensor
            3d tensor of shape (batch, channel, stats) containing the measured
            texture statistics.

        """
        if images.shape[0] != 1 and images.shape[0] != self.im_weights.shape[-1]:
            raise ValueError(
                f"input image must have 1 or {self.im_weights.shape[-1]} "
                f"elements on the batch dim, but has {images.shape[0]}!"
            )
        stats_vec_all = super().forward(images, scales=scales)
        # weighted sum across that first dimension
        stats_vec_all = torch.einsum("k, k c s -> c s", self.im_weights, stats_vec_all)
        # add back the first dim, as metamer needs 3d outputs of models
        return stats_vec_all.unsqueeze(0)

    def to(self, *args, **kwargs):
        r"""Moves and/or casts the parameters and buffers.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)

        .. function:: to(dtype, non_blocking=False)

        .. function:: to(tensor, non_blocking=False)

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired :attr:`dtype` s. In addition, this method will
        only cast the floating point parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.
        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module

        Returns:
            Module: self
        """
        super().to(*args, **kwargs)
        self.im_weights = self.im_weights.to(*args, **kwargs)
        return self


class MetamerMixture(po.synth.MetamerCTF):
    r"""Extending metamer synthesis, for mixing N images."""

    def _initialize(self, initial_image):
        """Initialize the metamer.

        Set the ``self.metamer`` attribute to be a parameter with
        the user-supplied data, making sure it's the right shape.

        Parameters
        ----------
        initial_image :
            The tensor we use to initialize the metamer. If None (the
            default), we initialize with uniformly-distributed random
            noise lying between 0 and 1.

        """
        if initial_image is None:
            initial_image = torch.rand_like(self.image[0].unsqueeze(0)) * 0.01 + self.image.mean()
        if initial_image.ndimension() < 4:
            raise Exception(
                "initial_image must be torch.Size([n_batch"
                ", n_channels, im_height, im_width]) but got "
                f"{initial_image.size()}"
            )
        # the difference between this and the regular version of Metamer is that
        # the regular version requires synthesized_signal and target_signal to have
        # the same shape, and here target_signal is (k, 1, h, w), not (1, 1, h, w)
        metamer = initial_image.clone().detach()
        metamer = metamer.to(dtype=self.image.dtype, device=self.image.device)
        metamer.requires_grad_()
        self._metamer = metamer


def load_metamer_mixture(file_path: str) -> MetamerMixture:
    """Load MetamerMixture object found at file.

    Note, this keeps everything on the cpu. To move it over to the GPU after
    loading, run ``metamer.to('gpu')``

    Parameters
    ----------
    file_path :
        Path to the ``metamer.pt`` object. We assume there's a single yml file
        in the directory with this file, which tells us how the model was
        initialized.

    Returns
    -------
    metamer :
        The MetamerMixture object saved at file_path

    """
    yml_file = glob(op.join(op.dirname(file_path), '*.yml'))
    if len(yml_file) != 1:
        raise Exception(f"We expect to find a single yml file in the same directory as {file_path}, but found {len(yml_file)} instead!")
    config = utils.read_yml(yml_file[0])
    imgs, weights = utils.prep_imgs(config['images_dict'], 'cpu')
    ps = PortillaSimoncelliMinimalMixture(imgs.shape[-2:], weights, **config['model_params'])
    ps.eval()
    met = MetamerMixture(
        imgs,
        ps,
        po.tools.optim.l2_norm,
        coarse_to_fine="together",
    )
    met.load(file_path)
    return met
