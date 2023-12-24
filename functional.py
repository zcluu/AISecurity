import torch
from typing import Iterable


def resume_img(
        img: torch.Tensor,
        dataset_mean: Iterable[float],
        dataset_std: Iterable[float],
):
    _img = list()
    for channel_id, (s, m) in enumerate(zip(dataset_std, dataset_mean)):
        channel = img[channel_id] * s + m
        _img.append(channel)
    img = torch.stack(_img, dim=0)
    img.clamp_(0, 1)
    return img


def load_checkpoint(model, ckpt_path, strict=False):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    ckpt = checkpoint['state_dict']
    ckpt_keys = ckpt.keys()
    ckpt = {ckpt_key.replace('module.', ''): ckpt[ckpt_key] for ckpt_key in ckpt_keys}
    model.load_state_dict(ckpt, strict=strict)
    return model
