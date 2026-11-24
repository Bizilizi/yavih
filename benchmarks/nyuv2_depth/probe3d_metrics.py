# Copied from Probe3D (El Banani et al., "Probing the 3D Awareness of Visual Foundation Models", CVPR 2024),
# https://github.com/mbanani/probe3d at commit a1f14640076e38b8bc07b66d0fe2d01d15691e9d, file evals/utils/metrics.py
# (sha256 39c0c919f0474b1a7b6d09eedba9dccda19abff921381343250c5058a44a34a8). MIT licence, notice below.
#
# Changes: only the two functions nyuv2_depth scores with are kept, byte for byte: evaluate_depth and the
# match_scale_and_shift it can call (scale_invariant=True; not used here). Dropped: depth_rmse and
# evaluate_surface_norm (not used) and `from loguru import logger` (only depth_rmse used it). The kept block's
# sha256 is PROBE3D_BLOCK_SHA256 in eval.py, so the folder test notices any edit.
#
# Imported by eval.py at scoring time only (torch is needed here; eval.py stays torch-free at module scope).
# ruff: noqa
"""
MIT License

Copyright (c) 2024 Mohamed El Banani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch


def evaluate_depth(
    depth_pr, depth_gt, image_average=False, scale_invariant=False, nyu_crop=False
):
    assert depth_pr.shape == depth_gt.shape, f"{depth_pr.shape} != {depth_gt.shape}"

    if len(depth_pr.shape) == 4:
        depth_pr = depth_pr.squeeze(1)
        depth_gt = depth_gt.squeeze(1)

    if nyu_crop:
        # apply NYU crop --- commonly used in many repos for some reason
        assert depth_pr.shape[-2] == 480
        assert depth_pr.shape[-1] == 640
        depth_pr = depth_pr[..., 45:471, 41:601]
        depth_gt = depth_gt[..., 45:471, 41:601]

    if scale_invariant:
        depth_pr = match_scale_and_shift(depth_pr, depth_gt)

    # zero out invalid pixels
    valid = (depth_gt > 0).detach().float()
    depth_pr = depth_pr * valid

    # get num valid
    num_valid = valid.sum(dim=(1, 2)).clamp(min=1)

    # get recall @ thresholds
    thresh = torch.maximum(
        depth_gt / depth_pr.clamp(min=1e-9), depth_pr / depth_gt.clamp(min=1e-9)
    )
    d1 = ((thresh < 1.25 ** 1).float() * valid).sum(dim=(1, 2)) / num_valid
    d2 = ((thresh < 1.25 ** 2).float() * valid).sum(dim=(1, 2)) / num_valid
    d3 = ((thresh < 1.25 ** 3).float() * valid).sum(dim=(1, 2)) / num_valid

    # compute RMSE
    sse = (depth_gt - depth_pr).pow(2)
    mse = (sse * valid).sum(dim=(1, 2)) / num_valid
    rmse = mse.sqrt()
    metrics = {"d1": d1.cpu(), "d2": d2.cpu(), "d3": d3.cpu(), "rmse": rmse.cpu()}

    if image_average:
        for key in metrics:
            metrics[key] = metrics[key].mean()

    return metrics


def match_scale_and_shift(prediction, target):
    # based on implementation from
    # https://gist.github.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0

    assert len(target.shape) == len(prediction.shape)
    if len(target.shape) == 4:
        four_chan = True
        target = target.squeeze(dim=1)
        prediction = prediction.squeeze(dim=1)
    else:
        four_chan = False

    mask = (target > 0).float()

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 *
    # a_10) . b
    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    # compute scale and shift
    scale = torch.ones_like(b_0)
    shift = torch.zeros_like(b_1)
    scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    scale = scale.view(-1, 1, 1).detach()
    shift = shift.view(-1, 1, 1).detach()
    prediction = prediction * scale + shift

    return prediction[:, None, :, :] if four_chan else prediction
