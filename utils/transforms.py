"""
@author: Yanzuo Lu
@email:  luyz5@mail2.sysu.edu.cn
"""

import torchvision.transforms as transforms
from timm.data.random_erasing import RandomErasing
from torchvision.transforms.functional import InterpolationMode

__all__ = [
    'mae_aug',
    'transreid_aug',
    'no_aug',
    'ratio_optimized'
]

def mae_aug(cfg):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(cfg.INPUT.IMAGE_SIZE, scale=(cfg.INPUT.MIN_SCALE, cfg.INPUT.MAX_SCALE), 
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def transreid_aug(cfg):
    transform = transforms.Compose([
        transforms.Resize(cfg.INPUT.IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(cfg.INPUT.PADDING),
        transforms.RandomCrop(cfg.INPUT.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu')
    ])
    return transform


def no_aug(cfg):
    transform = transforms.Compose([
        transforms.Resize(cfg.INPUT.IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def ratio_optimized(cfg):
    aspect_ratio = cfg.INPUT.IMAGE_SIZE[1] / cfg.INPUT.IMAGE_SIZE[0]
    transform = transforms.Compose([
        transforms.RandomResizedCrop(cfg.INPUT.IMAGE_SIZE, ratio=(aspect_ratio * 3. / 4., aspect_ratio * 4. / 3.), scale=(
                                     cfg.INPUT.MIN_SCALE, cfg.INPUT.MAX_SCALE), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def byol_aug(cfg):
    transform = transforms.Compose([
        transforms.Resize(cfg.INPUT.IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    return transform