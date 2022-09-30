"""
Different loss functions
Loss functions and seeting up loss function
"""

import torch
import torch.nn as nn
from CONFIG import LOSSES


class PixelLoss():
    """ Pixelwise reconstruction loss """

    def __init__(self, loss_type, class_weights=None):
        """ """
        self.loss_type = loss_type
        if loss_type not in LOSSES:
            raise ValueError(f"Unknown loss {loss_type}. Use one of {LOSSES}")
        if (loss_type == "mse"):
            self.criterion = nn.MSELoss()
        elif (loss_type == "mae"):
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def __call__(self, preds, targets, *args):
        """ Computing loss """
        if self.loss_type == "cross_entropy":
            targets_cls = torch.argmax(targets, dim=2)
            num_frames = preds.shape[1]
            loss = 0.
            for f in range(num_frames):
                loss += self.criterion(preds[:, f], targets_cls[:, f])
            loss = loss / num_frames
        else:
            loss = self.criterion(preds, targets)

        return loss


class KLLoss():
    """ Kullback-Leibler loss """

    def __call__(self, mu1, logvar1, mu2, logvar2):
        """
        Computing KL-loss for a minibatch

        Args:
        -----
        mu1, mu2, logvar1, logvar2: lists
            Lists of lists containing the mean and log-variances for the prior and posterior distributions,
            where each element is a tensor of shape (B, *latent_dim)
        """
        if (len(mu1) > 0 and (not isinstance(mu1[0], list) or len(mu1[0]) > 0)):
            if (isinstance(mu1[0], list)):  # HierarchModel case
                mu1, logvar1 = [torch.stack(m, dim=1) for m in mu1], [torch.stack(m, dim=1) for m in logvar1]
                mu2, logvar2 = [torch.stack(m, dim=1) for m in mu2], [torch.stack(m, dim=1) for m in logvar2]
                loss = 0.
                for m1, lv1, m2, lv2 in zip(mu1, logvar1, mu2, logvar2):
                    kld = self._kl_loss(m1, lv1, m2, lv2)
                    loss += kld.sum() / kld.shape[0]
            else:
                mu1, logvar1 = torch.stack(mu1, dim=1), torch.stack(logvar1, dim=1)  # stacking across Frame dim
                mu2, logvar2 = torch.stack(mu2, dim=1), torch.stack(logvar2, dim=1)  # stacking across Frame dim
                kld = self._kl_loss(mu1, logvar1, mu2, logvar2)
                loss = kld.sum() / kld.shape[0]
        else:
            loss = torch.tensor(0.)
        return loss

    def _kl_loss(self, mu1, logvar1, mu2, logvar2):
        """ Computing the KL-Divergence between two Gaussian distributions """
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
        return kld


class HeatmapLoss():
    """ Custom MSE loss for heatmaps of joints/keypoints """

    def __init__(self, channel_weights=[]):
        """ """
        self.criterion = nn.MSELoss()
        self.channel_weights = channel_weights

    def __call__(self, preds, targets, batch_classes):
        """
        Skip loss computations on heatmap channels with no corresponding GT-joint,
        unless there are no GT-joints for the given frame at all.

        Weight losses of individual joint heatmaps according to given @joint_weights_per_cls.
        """
        B, F, C, _, _ = preds.shape
        loss = 0.
        N = 0
        for b in range(B):
            channel_weights = self.channel_weights
            if batch_classes != []:
                channel_weights = channel_weights[batch_classes[b]]
            for f in range(F):
                target_hmaps = targets[b, f]
                empty_target = (torch.count_nonzero(target_hmaps) == 0)
                for c in range(C):
                    if empty_target or torch.count_nonzero(target_hmaps[c]) > 0:
                        w = channel_weights[c]
                        loss += w * self.criterion(preds[b, f, c], target_hmaps[c])
                        N += 1
        loss = loss / N
        return loss


#
