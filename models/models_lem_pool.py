# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

__all__ = [
    'lem_pool_base_patch16',
    'lem_pool_large_patch16',
    'lem_pool_huge_patch14'
]

class LocalityEnhancedModule(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(LocalityEnhancedModule, self).__init__(**kwargs)

        self.pool_norm = kwargs['norm_layer'](kwargs['embed_dim'])
        self.pool_fc = nn.Linear(self.num_features, kwargs['num_classes']) if kwargs['num_classes'] > 0 else nn.Identity()
        self.pool_bn = nn.BatchNorm1d(kwargs['embed_dim'])
        self.pool_bn.bias.requires_grad_(False)

        self.bn = nn.BatchNorm1d(kwargs['embed_dim'])
        self.bn.bias.requires_grad_(False)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        _x = x[:, 1:].mean(dim=1)
        _x = self.pool_norm(_x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = (self.fc_norm(x), _x)
        else:
            x = self.norm(x)
            outcome = (x[:, 0], _x)

        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            logit = (self.head(self.bn(x[0])), self.pool_fc(self.pool_bn(x[1])))
        return x, logit


def lem_pool_base_patch16(cfg, num_classes):
    model = LocalityEnhancedModule(img_size=cfg.INPUT.IMAGE_SIZE, num_classes=num_classes,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=cfg.MODEL.DROP_PATH)
    return model


def lem_pool_large_patch16(cfg, num_classes):
    model = LocalityEnhancedModule(img_size=cfg.INPUT.IMAGE_SIZE, num_classes=num_classes,
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=cfg.MODEL.DROP_PATH)
    return model


def lem_pool_huge_patch14(cfg, num_classes):
    model = LocalityEnhancedModule(img_size=cfg.INPUT.IMAGE_SIZE, num_classes=num_classes,
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=cfg.MODEL.DROP_PATH)
    return model
