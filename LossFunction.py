import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from LovaszSoftmax import lovasz_softmax


def loss_MAE(p, p_pred):
    _, _, width, height = p.shape
    return torch.sum(p - p_pred) / (width * height)


def loss_SSIM(p, p_pred, data_range=255):
    return 1 - ssim(p, p_pred, data_range=data_range, size_average=True)


def loss_LZS(i, i_pred):
    return lovasz_softmax(i_pred, i)


def loss(p, p_pred, data_range, i, i_pred, lambda1, lambda2, lambda3):
    return lambda1 * loss_MAE(p, p_pred) + lambda2 * loss_SSIM(p, p_pred, data_range) + lambda3 * loss_LZS(i, i_pred)
