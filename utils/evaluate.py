import numpy as np
from skimage.metrics import structural_similarity
import lpips
import torch
import math


def mse(imgs1, imgs2):
    # [batch_size, 1, h, w]
    # assume img in range [0, 1]
    if imgs1.ndim==4:
        imgs1 = np.squeeze(imgs1, axis=1)
        imgs2 = np.squeeze(imgs2, axis=1)
    mse = np.mean( (imgs1/1.0 - imgs2/1.0) ** 2 )
    return mse


def psnr(imgs1, imgs2):
    # [batch_size, 1, h, w]
    # assume img in range [0, 1]
    if imgs1.ndim==4:
        imgs1 = np.squeeze(imgs1, axis=1)
        imgs2 = np.squeeze(imgs2, axis=1)
    mse = np.mean( (imgs1/1.0 - imgs2/1.0) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def ssim(imgs1, imgs2):
    if imgs1.ndim==4:
        imgs1 = np.squeeze(imgs1, axis=1)
        imgs2 = np.squeeze(imgs2, axis=1)
    all_ssim = 0
    batch_size = np.size(imgs1, 0)
    for i in range(batch_size):
        cur_ssim = structural_similarity(np.squeeze(imgs1[i]), np.squeeze(imgs2[i]),\
            multichannel=False, data_range=1.0)
        all_ssim += cur_ssim
    final_ssim = all_ssim / batch_size
    return final_ssim


class PerceptualLoss:
    def __init__(self, net='vgg', device='cuda:0'):
        """
        Wrapper for PerceptualSimilarity.models.PerceptualLoss
        """
        self.model = lpips.LPIPS(net=net).to(device)

    def __call__(self, pred, target, normalize=True):
        """
        pred and target are Tensors with shape N x C x H x W (C {1, 3})
        normalize scales images from [0, 1] to [-1, 1] (default: True)
        PerceptualLoss expects N x 3 x H x W.
        """
        if pred.shape[1] == 1:
            pred = torch.cat([pred, pred, pred], dim=1)
        if target.shape[1] == 1:
            target = torch.cat([target, target, target], dim=1)
        dist = self.model.forward(pred, target, normalize=normalize)
        return dist.mean()

