"""Per-sample losses (the pinned ``loss`` names; exact per-sample-sum accumulation).

Contract: every loss maps ``(pred, target)`` to a ``[B]`` vector and
the TRAINER reduces it as ``per_sample.sum() / batch``. The micro-batch sums then add up to the
whole-batch numerator exactly, for GIoU and log-depth as much as for CE. Invalid targets are ``NaN``
(float) or ``< 0`` (integer index), masked INSIDE each loss and never allowed into the arithmetic
(``0 * NaN`` is ``NaN``)."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import functional as F

__all__ = ["per_sample_loss", "LOSSES", "DEPTH_EPS", "generalized_iou", "cxcywh_to_xyxy"]

#: Floor for a depth before ``log``; the clamp zeroes the gradient of anything that falls through it.
DEPTH_EPS = 1e-6
_BOX_EPS = 1e-7


def _align(pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Drop a trailing singleton so ``[B, S, S, 1]`` meets ``[B, S, S]``; any other mismatch raises."""
    if pred.dim() == target.dim() + 1 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    elif target.dim() == pred.dim() + 1 and target.shape[-1] == 1:
        target = target.squeeze(-1)
    if pred.shape != target.shape:
        raise ValueError(f"loss shape mismatch: prediction {tuple(pred.shape)} vs target {tuple(target.shape)}")
    return pred, target


def _valid_float(target: Tensor) -> tuple[Tensor, Tensor]:
    """``(safe, valid)``: non-finite entries replaced by 0 so they never reach the arithmetic."""
    valid = torch.isfinite(target)
    if bool(valid.all()):
        return target, valid
    return torch.where(valid, target, torch.zeros_like(target)), valid


def _masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    """Per-sample mean over valid entries → ``[B]``; a sample with no valid entry contributes exactly 0."""
    b = values.shape[0]
    flat = values.reshape(b, -1)
    mask = valid.reshape(b, -1).to(flat.dtype)
    return (flat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def _class_indices(target: Tensor, num_classes: int) -> tuple[Tensor, Tensor]:
    """``(safe int64, valid)``: NaN (float) or negative (int) entries are invalid."""
    if target.is_floating_point():
        valid = torch.isfinite(target) & (target >= 0)
        safe = torch.where(valid, target, torch.zeros_like(target)).to(torch.long)
    else:
        target = target.to(torch.long)
        valid = target >= 0
        safe = torch.where(valid, target, torch.zeros_like(target))
    return safe.clamp(0, max(0, num_classes - 1)), valid


def ce(pred: Tensor, target: Tensor) -> Tensor:
    """Single-label CE. ``pred [B, C]``, ``target [B]`` (int64, or float with NaN = invalid)."""
    if pred.dim() != 2:
        raise ValueError(f"ce expects [B, C] logits, got {tuple(pred.shape)}")
    target = target.reshape(pred.shape[0]) if target.dim() > 1 else target
    idx, valid = _class_indices(target, pred.shape[-1])
    per = F.cross_entropy(pred, idx, reduction="none")
    return per * valid.to(per.dtype)


def bce(pred: Tensor, target: Tensor) -> Tensor:
    """Multi-label BCE-with-logits, mean over the valid classes of each row. ``[B, C]`` both."""
    pred, target = _align(pred, target)
    safe, valid = _valid_float(target.to(pred.dtype))
    return _masked_mean(F.binary_cross_entropy_with_logits(pred, safe, reduction="none"), valid)


def ce_seq(pred: Tensor, target: Tensor) -> Tensor:
    """Per-segment CE, NaN-masked (seq). ``pred [B, T, C]``, ``target [B, T]``."""
    if pred.dim() != 3:
        raise ValueError(f"ce_seq expects [B, T, C] logits, got {tuple(pred.shape)}")
    b, t, c = pred.shape
    if tuple(target.shape[:2]) != (b, t):
        raise ValueError(f"ce_seq target must be [B, T] = {(b, t)}, got {tuple(target.shape)}")
    idx, valid = _class_indices(target.reshape(b * t), c)
    per = F.cross_entropy(pred.reshape(b * t, c), idx, reduction="none")
    return _masked_mean(per.reshape(b, t), valid.reshape(b, t))


def l1_log(pred: Tensor, target: Tensor) -> Tensor:
    """L1 in log-depth over valid pixels. ``pred`` is a DEPTH (the grid head applies
    ``exp``); invalid target pixels are ``NaN`` or ``<= 0`` (the readout NaN-masks ``d > max_depth``)."""
    pred, target = _align(pred, target)
    valid = torch.isfinite(target) & (target > 0)
    safe_t = torch.where(valid, target, torch.ones_like(target)).to(pred.dtype)
    diff = torch.log(pred.clamp_min(DEPTH_EPS)) - torch.log(safe_t)
    return _masked_mean(diff.abs(), valid)


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(dim=-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_iou(pred_xyxy: Tensor, target_xyxy: Tensor) -> Tensor:
    """Row-aligned GIoU of two ``[..., 4]`` xyxy sets."""
    px1, py1, px2, py2 = pred_xyxy.unbind(dim=-1)
    tx1, ty1, tx2, ty2 = target_xyxy.unbind(dim=-1)
    area_p = (px2 - px1).clamp_min(0) * (py2 - py1).clamp_min(0)
    area_t = (tx2 - tx1).clamp_min(0) * (ty2 - ty1).clamp_min(0)
    inter = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp_min(0) * (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp_min(0)
    union = area_p + area_t - inter
    iou = inter / union.clamp_min(_BOX_EPS)
    enclose = (torch.max(px2, tx2) - torch.min(px1, tx1)).clamp_min(0) * (torch.max(py2, ty2) - torch.min(py1, ty1)).clamp_min(0)
    return iou - (enclose - union) / enclose.clamp_min(_BOX_EPS)


def box_l1_giou(pred: Tensor, target: Tensor) -> Tensor:
    """Mean L1 over the 4 ``cxcywh`` coordinates + ``(1 − GIoU)`` (box). A row with any
    non-finite target coordinate contributes zero loss and zero gradient."""
    pred, target = _align(pred, target)
    if pred.shape[-1] != 4:
        raise ValueError(f"box_l1_giou expects [B, 4] boxes, got {tuple(pred.shape)}")
    valid = torch.isfinite(target).all(dim=-1)
    safe = torch.where(valid.unsqueeze(-1), target, torch.zeros_like(target)).to(pred.dtype)
    l1 = (pred - safe).abs().mean(dim=-1)
    giou = generalized_iou(cxcywh_to_xyxy(pred), cxcywh_to_xyxy(safe))
    loss = l1 + (1.0 - giou)
    return loss * valid.to(loss.dtype)


_LOSSES: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "ce": ce,
    "bce": bce,
    "ce_seq": ce_seq,
    "l1_log": l1_log,
    "box_l1_giou": box_l1_giou,
}
LOSSES: tuple[str, ...] = tuple(_LOSSES)


def per_sample_loss(name: str) -> Callable[[Tensor, Tensor], Tensor]:
    """``(pred, target) -> [B]`` for a pinned loss name."""
    if name not in _LOSSES:
        raise ValueError(f"unknown loss {name!r}; expected one of {LOSSES}")
    return _LOSSES[name]
