# Python standard library imports
import warnings
from itertools import product
from typing import Callable, Optional, Union

# PyTorch related imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
# MONAI related imports
from monai.networks import one_hot
from monai.utils import LossReduction, convert_data_type, optional_import
# Other third-party libraries
import cv2
import numpy as np


ALPHA = 0.8
GAMMA = 2


distance_transform_edt, _ = optional_import("scipy.ndimage", name="distance_transform_edt")

class ShapeDistLoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        smooth_nr: float = 1e-8,
        smooth_k: float = 2e-1,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.smooth_nr = float(smooth_nr)
        self.smooth_k = float(smooth_k)

    def distance_map(self, mask: torch.Tensor) -> torch.Tensor:
        # Convert to NumPy to use with SciPy
        roi, _, _ = convert_data_type(mask, np.ndarray)

        # Compute normalized distance transform
        dt: np.ndarray = distance_transform_edt(roi)
        dt /= dt.max() + self.smooth_nr

        # apply Heaviside function to softly normalize into [0, 1]
        result: np.ndarray = 1 / (1 + np.exp(-(1 - dt) / self.smooth_k))
        # mask using region of interest
        result *= roi

        return torch.Tensor(result)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                pass
                #warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        distance_maps = torch.empty(size=target.size()).to(input.device)

        for im, ch in product(*map(range, target.shape[:2])):
            distance_maps[im, ch] = self.distance_map(target[im, ch])

        f = (distance_maps - input).abs().sum(dim=(2, 3)) / input.sum(dim=(2, 3))

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f
    
class BoundaryLoss(nn.Module):

    def __init__(self) -> None:
        """
        compute boudary loss
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super(BoundaryLoss, self).__init__()
        
    def forward(self, outputs, gt):
        """
        net_output: (batch_size, class, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        bound: precomputed distance map, shape (batch_size, class, x,y,z)
        """
        n_pred_ch = input.shape[1]
        target = one_hot(target, num_classes=n_pred_ch)
        net_output = F.softmax(outputs, dim=1)
        
        pc = net_output[:, 1:, ...].type(torch.float32)
        dc = gt[:,1:, ...].type(torch.float32)
        multipled = torch.einsum("bcxyz,bcxyz->bcxyz", pc, dc)
        bd_loss = multipled.mean()
        
        return bd_loss
    
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs.float(), targets.float(), reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

        return focal_loss