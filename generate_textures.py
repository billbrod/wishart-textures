#!/usr/bin/env python3

import torch
import numpy as np
import plenoptic as po
from collections import OrderedDict


class PortillaSimoncelliMinimalMixture(po.simul.PortillaSimoncelli):
    r"""Model for obtaining the minimal set of PS statistics on mixtures of image

    This class returns the minimal set of PS statistics, removing some
    redundant and unreported statistics from the set returned by the full
    plenoptic.simulate.PortillaSimoncelli object, returning ~700 instead of
    ~1700 for ``n_scales=4``, ``n_orientations=4``, and
    ``spatial_corr_width=7``. Synthesis performed with this model will be
    identical to the full one.

    Additionally, this model can accept multiple images, indexed on the batch
    dimension (remember, inputs have shape (batch, channel, height, width)).
    The model is initialized with a weight vector, which tells us how to weight
    the outputs of these images, so as to produce a mixture image. It can also
    accept a single image, in which case it behaves as normal.

    This is based off the ``PortillaSimoncelliMinimalStats`` and
    ``PortillaSimoncelliMixture`` classes found at the end of the PS texture
    metamer notebook, but here we remove the redundant statistics entirely,
    instead of just zeroing them out and we allow for an arbitrary number of
    images, with variable weighting

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
    use_true_correlations: bool
        if True, use the true correlations, otherwise use covariances

    """
    def __init__(
        self,
        im_shape,
        im_weights,
        n_scales=4,
        n_orientations=4,
        spatial_corr_width=7,
        use_true_correlations=True,
    ):
        super().__init__(im_shape, n_scales=n_scales,
                         n_orientations=n_orientations,
                         spatial_corr_width=spatial_corr_width,
                         use_true_correlations=use_true_correlations)
        self.im_weights = im_weights
        # Turn the mask dictionary into a vector
        statistics_mask = self.mask_extra_statistics()
        try:
            # this works before the refactor of the PS code that allows it to
            # accept multi-batch and multi-channel images
            self.statistics_mask = self.convert_to_vector(statistics_mask)
        except IndexError:
            # and this works after
            new_stats_mask = OrderedDict()
            for k, v in statistics_mask.items():
                new_stats_mask[k] = v.unsqueeze(0).unsqueeze(0)
            self.statistics_mask = self.convert_to_vector(new_stats_mask)
        # make representation_scales the same size
        self.representation_scales = np.array(self.representation_scales)[self.statistics_mask.squeeze()]
       # need to make sure the numbers are ints
        rep_scales = []
        for i in self.representation_scales:
            try:
                rep_scales.append(int(i))
            except ValueError:
                rep_scales.append(i)
        self.representation_scales = rep_scales
        # pytorch doesn't like boolean indexing...
        self.statistics_mask = torch.where(self.statistics_mask)[-1]

    def mask_extra_statistics(self):
        r"""Generate a dictionary with the same structure as the statistics
        dictionary, containing masks that indicate for each statistics
        whether it is part of the minimal set of original statistics (True)
        or not (False).
        """
        n = self.spatial_corr_width
        n_scales = self.n_scales
        n_orientations = self.n_orientations
        mask_original = OrderedDict()  # Masks of original statistics
        # Add mask elements in same order as the po statistics dict

        #### pixel_statistics ####
        # All in original statistics
        mask_original['pixel_statistics'] = torch.tensor([True] * 6)

        #### magnitude_means ####
        # Not in original paper
        mask_original['magnitude_means'] = torch.tensor([False] * ((n_scales * n_orientations) + 2))

        #### auto_correlation_magnitude ####
        # Symmetry M_{i,j} = M_{n-i+1, n_j+1}
        # Start with 0's and put 1's in original elements
        acm_mask = torch.zeros((n, n, n_scales, n_orientations))
        # Lower triangular (including diagonal) to ones
        tril_inds = torch.tril_indices(n, n)
        acm_mask[tril_inds[0], tril_inds[1], :, :] = 1
        # Set repeated diagnoal elements to 0
        diag_repeated = torch.arange(start=(n+1)/2, end=n, dtype=torch.long)
        acm_mask[diag_repeated, diag_repeated, :, :] = 0
        mask_original['auto_correlation_magnitude'] = acm_mask.bool()

        #### skew_reconstructed, kurtosis_reconstructed ####
        # All in original paper
        mask_original['skew_reconstructed'] = torch.tensor([True] * (n_scales + 1))
        mask_original['kurtosis_reconstructed'] = torch.tensor([True] * (n_scales + 1))

        #### auto_correlation_reconstructed ####
        # Symmetry M_{i,j} = M_{n-i+1, n-j+1}
        acr_mask = torch.zeros((n, n, n_scales+1))
        # Reuse templates from acm
        acr_mask[tril_inds[0], tril_inds[1], :] = 1
        acr_mask[diag_repeated, diag_repeated, :] = 0
        mask_original['auto_correlation_reconstructed'] = acr_mask.bool()
        if self.use_true_correlations:
            # std_reconstructed holds the center values of the
            # auto_correlation_reconstructed matrices. Which are turned
            # to 1's when using correlations
            mask_original['std_reconstructed'] = torch.tensor([True] * (n_scales + 1))

        #### cross_orientation_correlation magnitude ####
        # Symmetry M_{i,j} = M_{j,i}. Diagonal elements are redundant with the
        # central elements of acm matrices. Last scale is full of 0's
        # Start with 1's and set redundant elements to 0
        cocm_mask = torch.ones((n_orientations, n_orientations, n_scales+1))
        # Template of redundant indices (diagonals are redundant)
        triu_inds = torch.triu_indices(n_orientations, n_orientations)
        cocm_mask[triu_inds[0], triu_inds[1], :] = 0
        # Set to 0 last scale that is not in the paper
        cocm_mask[:, :, -1] = 0
        mask_original['cross_orientation_correlation_magnitude'] = cocm_mask.bool()

        #### cross_scale_correlation_magnitude ####
        # No symmetry. Last scale is always 0
        cscm_mask = torch.ones((n_orientations, n_orientations, n_scales))
        cscm_mask[:,:,-1] = 0
        mask_original['cross_scale_correlation_magnitude'] = cscm_mask.bool()

        #### cross_orientation_correlation_real ####
        # Not included in paper's statistics
        mask_original['cross_orientation_correlation_real'] = torch.zeros((n_orientations*2, n_orientations*2, n_scales+1)).bool()

        #### cross_scale_correlation_real ####
        # No symmetry. Bottom half of matrices are 0's always.
        # Last scale is not included in paper's statistics
        cscr_mask = torch.ones((n_orientations*2, n_orientations*2, n_scales))
        cscr_mask[(n_orientations):, :, :] = 0
        cscr_mask[:, :, (n_scales-1):] = 0
        mask_original['cross_scale_correlation_real'] = cscr_mask.bool()

        ### var highpass residual ####
        # Not redundant
        mask_original['var_highpass_residual'] = torch.tensor(True)

        ### Adjust dictionary for correlation matrices ####
        if self.use_true_correlations:
            # Constant 1's in the correlation matrices not in original set
            ctrind = torch.tensor([n//2])
            mask_original['auto_correlation_reconstructed'][ctrind, ctrind, :] = False
            mask_original['auto_correlation_magnitude'][ctrind, ctrind, :, :] = False
            # Remove from original set the diagonal elements of
            # cross_orientation_correlation_magnitude matrices
            # that are 1's in correlation matrices
            dgind = torch.arange(n_orientations)
            mask_original['cross_orientation_correlation_magnitude'][dgind, dgind, :-1] = True
        return mask_original


    def forward(self, image, scales=None):
        r"""Generate the minimal, mixed texture representation.

        Parameters
        ----------
        image : torch.Tensor
            A 4d tensor containing the image(s) to analyze, with shape (k,
            channel, height, width), where i in {1, self.im_weights.shape[-1]}
        scales : list, optional
            Which scales to include in the returned representation. If an empty
            list (the default), we include all scales. Otherwise, can contain
            subset of values present in this model's ``scales`` attribute.

        Returns
        -------
        representation_vector: torch.Tensor
            A flattened tensor (1d) containing the measured representation statistics.

        """
        if image.shape[0] != 1 and image.shape[0] != self.im_weights.shape[-1]:
            raise ValueError(f"input image must have 1 or {self.im_weights.shape[-1]} "
                             f"elements on the batch dim, but has {image.shape[0]}!")
        stats_vec_all = []
        for img in image:
            # create the representation vector with (with all scales)
            stats_vec = super().forward(img.unsqueeze(0))
            # Remove the redundant stats
            stats_vec = stats_vec.index_select(-1, self.statistics_mask)
            # then remove any scales we don't want (for coarse-to-fine)
            if scales is not None:
                stats_vec = self.remove_scales(stats_vec, scales)
            stats_vec_all.append(stats_vec)
        stats_vec_all = torch.cat(stats_vec_all)
        # weighted sum across that first dimension
        stats_vec_all = torch.einsum('k, k c s -> c s', self.im_weights, stats_vec_all)
        # add back the first dim, as metamer needs 3d outputs of models
        return stats_vec_all.unsqueeze(0)


class MetamerMixture(po.synth.MetamerCTF):
    r""" Extending metamer synthesis, for mixing N images.
    """

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
        if initial_image.ndimension() < 4:
            raise Exception("initial_image must be torch.Size([n_batch"
                            ", n_channels, im_height, im_width]) but got "
                            f"{initial_image.size()}")
        # the difference between this and the regular version of Metamer is that
        # the regular version requires synthesized_signal and target_signal to have
        # the same shape, and here target_signal is (k, 1, h, w), not (1, 1, h, w)
        metamer = initial_image.clone().detach()
        metamer = metamer.to(dtype=self.image.dtype,
                             device=self.image.device)
        metamer.requires_grad_()
        self._metamer = metamer
