# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#  Edited: Yuekun Dai, Siyao Li
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import argparse
import matplotlib.pyplot as plt
import open_clip
import os
import torch
import torch.nn.functional as F
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from torch import nn
from torch_scatter import scatter as super_pixel_pooling

from basicsr.utils.registry import ARCH_REGISTRY
from raft.raft import RAFT


def MLP(channels: list, do_bn=True):
    """Multi-layer perceptron"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * width, one * height, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    # print(kpts.size(), center[:, None, :].size(), scaling[:, None, :].size())
    return (kpts - center[:, None, :]) / scaling[:, None, :]


def img2boxseq(image, keypoints, segment, scale_list=[1], box_size=64, interpolate="bilinear"):
    h, w = image.shape[-2:]
    boxes = [[] for _ in scale_list]

    for idx, box in enumerate(keypoints[0]):
        mask = segment == (idx + 1)
        catted = torch.cat([image, mask], dim=-3)
        x_min, x_max, y_min, y_max = box.tolist()
        d = max(x_max - x_min + 1, y_max - y_min + 1)
        r = (d + 1) // 2
        r = max(r, (box_size + 1) // 2)
        rs = [r * scale for scale in scale_list]
        x_center, y_center = (x_min + x_max) // 2, (y_min + y_max) // 2
        for i, r in enumerate(rs):
            x_min, x_max = x_center - r, x_center + r
            y_min, y_max = y_center - r, y_center + r
            x_offset = max(0 - x_min, 0) + min(w - x_max, 0)
            y_offset = max(0 - y_min, 0) + min(h - y_max, 0)
            x_min, x_max = x_min + x_offset, x_max + x_offset
            y_min, y_max = y_min + y_offset, y_max + y_offset
            cropped = catted[:, :, y_min : y_max + 1, x_min : x_max + 1]
            resized = F.interpolate(cropped.float(), (box_size, box_size), mode=interpolate)
            boxes[i].append(resized)

    stacked = [torch.stack(x, dim=1) for x in boxes]  # x -> n, s, c, h, w
    box_seq = torch.cat(stacked, dim=2)  # 1, #seg, #scale * c, h, w
    return box_seq


def flow_warp(x, flow, interpolation="bilinear", padding_mode="zeros", align_corners=True):
    """Warp an image or a feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f"The spatial sizes of input ({x.size()[-2:]}) and " f"flow ({flow.size()[1:3]}) are not the same.")
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(x, grid_flow, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)
    return output


class RaftWarper(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.raft = RAFT(args)
        state_dict = torch.load(args["ckpt"])
        real_state_dict = {k.split("module.")[-1]: v for k, v in state_dict.items()}
        self.raft.load_state_dict(real_state_dict)
        if args["freeze"]:
            for param in self.raft.parameters():
                param.requires_grad = False

    def forward(self, line, line_ref, color_ref):
        h, w = line.shape[-2:]
        line = F.interpolate(line, (320, 320), mode="bilinear", align_corners=False)
        line_ref = F.interpolate(line_ref, (320, 320), mode="bilinear", align_corners=False)
        color_ref = F.interpolate(color_ref, (320, 320), mode="bilinear", align_corners=False)
        self.raft.eval()
        _, flow_up = self.raft(line, line_ref, iters=20, test_mode=True)
        warped_img = flow_warp(color_ref, flow_up.permute(0, 2, 3, 1).detach(), "nearest")
        warped_img = F.interpolate(warped_img, (h, w), mode="bilinear", align_corners=False)
        return warped_img


class CLIPEncoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        clip_encoder, _, self.preprocess = open_clip.create_model_and_transforms("convnext_large_d_320", pretrained="laion2b_s29b_b131k_ft_soup")
        # We just assume preprocess is the same as our proprocess
        visual_model = clip_encoder.visual
        visual_model = list(visual_model.children())[0].children()
        visual_model0, visual_model1 = list(visual_model)[:-2]
        self.visual_encoder = torch.nn.Sequential(visual_model0, *list(visual_model1)[:-2])
        # For the visual encoder, [:-3] means [192,80,80] (default), [:-2] means [384,40,40] and [:-1] means [768,20,20]
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.visual_encoder(x)
        return x.detach()


class ClipFeatureExtracter(nn.Module):
    def __init__(self, input_channels, enc_dim):
        super().__init__()
        self.input_channels = input_channels
        self.clip = CLIPEncoder(freeze=True)
        self.fc = nn.Linear(384, enc_dim)

    def forward(self, x, seg):
        n, c, h, w = x.size()
        x = F.interpolate(x, (320, 320), mode="bilinear", align_corners=False)
        x = self.clip(x)
        x = F.interpolate(x, (h, w), mode="bilinear", align_corners=False)
        x = super_pixel_pooling(x.view(n, 384, -1), seg.view(-1).long(), reduce="mean")
        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class SegmentEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, input_size, input_channels, enc_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, enc_dim // 4, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(enc_dim // 4, enc_dim // 2, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(enc_dim // 2, enc_dim, 3, padding=1)

        self.norm1 = nn.InstanceNorm2d(enc_dim // 4)
        self.norm2 = nn.InstanceNorm2d(enc_dim // 2)
        self.norm3 = nn.InstanceNorm2d(enc_dim)

        self.finalpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_input_size = enc_dim * 4 * 4
        self.fc = nn.Linear(self.fc_input_size, enc_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, img):
        x = self.pool1(F.relu(self.norm1(self.conv1(img))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = self.norm3(self.conv3(x))
        x = self.finalpool(x)
        x = self.fc(x.view(-1, self.fc_input_size))
        return x


class SegmentDescriptor(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, input_size, input_channels, enc_dim):
        super().__init__()
        self.encoder = SegmentEncoder(input_size, input_channels, enc_dim)

    def forward(self, x):
        n, s, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self.encoder(x)  # n*s * d
        return x.view(n, s, -1).permute(0, 2, 1)


class KeypointEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([4] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        inputs = kpts.transpose(1, 2)
        # print(inputs.size(), 'wula!')
        x = self.encoder(inputs)
        # print(x.size())
        return x


def attention(query, key, value, mask=None):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim**0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(nn.Module):
    """Multi-head attention to increase model expressivitiy"""

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value, mask=None):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1) for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value, mask)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, mask=None):
        message = self.attn(x, source, source, mask)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([AttentionalPropagation(feature_dim, 4) for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == "cross":
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def transport(scores, alpha):
    """Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    # pad additional scores for unmatcheed (to -1)
    # alpha is the learned threshold
    couplings = torch.cat([torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1)

    return couplings


@ARCH_REGISTRY.register()
class BasicPBC_light(nn.Module):
    """SuperGlue feature matching middle-end. A new hard-coded self-attention will be added to the transformer.
    This part is an AnT module with the hard coded transformer.

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    This version also adds a hard-coded transformer.

    """

    def __init__(self, descriptor_dim=128, keypoint_encoder=[32, 64, 128], GNN_layer_num=6, token_scale_list=[1], token_crop_size=64, use_clip=False):

        super().__init__()

        config = argparse.Namespace()
        config.descriptor_dim = descriptor_dim
        config.keypoint_encoder = keypoint_encoder
        config.GNN_layers_num = GNN_layer_num
        config.GNN_layers = ["self", "cross"] * GNN_layer_num
        config.token_scale_list = token_scale_list
        config.token_crop_size = token_crop_size
        config.use_clip = use_clip

        self.config = config

        self.kenc = KeypointEncoder(self.config.descriptor_dim, self.config.keypoint_encoder)

        self.gnn = AttentionalGNN(self.config.descriptor_dim, self.config.GNN_layers)

        self.final_proj = nn.Conv1d(self.config.descriptor_dim, self.config.descriptor_dim, kernel_size=1, bias=True)

        if config.use_clip:
            self.clip = ClipFeatureExtracter(5 * len(self.config.token_scale_list), self.config.descriptor_dim)
            self.fuse = nn.Linear(2 * self.config.descriptor_dim, self.config.descriptor_dim)

        args = {
            "mixed_precision": False,
            "small": False,
            "ckpt": "raft/ckpt/raft-animerun-v2-ft_again.pth",
            "freeze": True,
        }
        self.raft_warper = RaftWarper(args)

        bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter("bin_score", bin_score)
        self.segment_desc = SegmentDescriptor(self.config.token_crop_size, 5 * len(self.config.token_scale_list), self.config.descriptor_dim)

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""

        warpped_img = self.raft_warper(data["line"], data["line_ref"], data["recolorized_img"])

        warpped_target_img = torch.cat((warpped_img, torch.mean(data["line"], dim=-3, keepdim=True)), dim=-3)
        warpped_ref_img = torch.cat((data["recolorized_img"], torch.mean(data["line_ref"], dim=-3, keepdim=True)), dim=-3)

        input_seq = img2boxseq(warpped_target_img, data["keypoints"], data["segment"], self.config.token_scale_list, self.config.token_crop_size)
        input_seq_ref = img2boxseq(warpped_ref_img, data["keypoints_ref"], data["segment_ref"], self.config.token_scale_list, self.config.token_crop_size)

        desc, desc_ref = self.segment_desc(input_seq), self.segment_desc(input_seq_ref)
        kpts, kpts_ref = data["keypoints"].float(), data["keypoints_ref"].float()

        if kpts.shape[1] < 2 or kpts_ref.shape[1] < 2:  # no keypoints
            shape0, shape1 = kpts.shape[:-1], kpts_ref.shape[:-1]
            print(data["file_name"])
            return {"matches0": kpts.new_full(shape0, -1, dtype=torch.int)[0], "matching_scores0": kpts.new_zeros(shape0)[0], "skip_train": True}

        all_matches = data["all_matches"] if "all_matches" in data else None  # shape = (1, K1)

        # positional embedding
        # Keypoint normalization.
        kpts = normalize_keypoints(kpts, data["line"].shape)
        kpts_ref = normalize_keypoints(kpts_ref, data["line_ref"].shape)

        # Keypoint MLP encoder.
        pos = self.kenc(kpts)
        pos_ref = self.kenc(kpts_ref)

        desc = desc + pos
        desc_ref = desc_ref + pos_ref

        if self.config.use_clip:
            clip_fea = self.clip(data["line"], data["segment"])[..., 1:]
            clip_fea_ref = self.clip(data["line_ref"], data["segment_ref"])[..., 1:]
            desc = self.fuse(torch.cat([desc, clip_fea], dim=1).permute(0, 2, 1)).permute(0, 2, 1)
            desc_ref = self.fuse(torch.cat([desc_ref, clip_fea_ref], dim=1).permute(0, 2, 1)).permute(0, 2, 1)

        # Multi-layer Transformer network.
        desc, desc_ref = self.gnn(desc, desc_ref)

        # Final MLP projection.
        mdesc, mdesc_ref = self.final_proj(desc), self.final_proj(desc_ref)

        # Compute matching descriptor distance.
        scores = torch.einsum("bdn,bdm->bnm", mdesc, mdesc_ref)

        # b k1 k2
        scores = scores / self.config.descriptor_dim**0.5

        # Run the optimal transport.
        b, m, n = scores.size()

        scores = transport(scores, self.bin_score)

        weights = data["numpixels"].float()  # .cuda()
        all_matches_origin = all_matches.clone() if all_matches is not None else None

        if all_matches is not None:
            all_matches[all_matches == -1] = n
            loss = nn.functional.cross_entropy(scores[:, :-1, :].view(-1, n + 1), all_matches.long().view(-1), reduction="mean")
            loss = loss.mean()

        scores = nn.functional.softmax(scores, dim=2)

        max0, max1 = scores[:, :-1, :].max(2), scores[:, :, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mscores0 = max0.values

        valid0 = indices0 < n
        valid1 = indices1 < m
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        if all_matches is not None:
            return {
                "match_scores": scores[:, :-1, :][0],
                "matches0": indices0[0],  # use -1 for invalid match
                "matching_scores0": mscores0[0],
                "loss": loss,
                "skip_train": False,
                "accuracy": ((all_matches_origin[0] == indices0[0]).sum() / len(all_matches_origin[0])).item(),
                "area_accuracy": (
                    torch.tensor(
                        [(data["segment"] == ii + 1).sum() for ii in torch.arange(0, all_matches_origin[0].shape[0])[all_matches_origin[0] == indices0[0]]]
                    ).sum()
                    / (weights.sum() * 1.0)
                ).item(),
                "valid_accuracy": (((all_matches_origin[0] == indices0[0]) & (all_matches_origin[0] != -1)).sum() / (all_matches_origin[0] != -1).sum()).item(),
                "invalid_accuracy": (
                    (((all_matches_origin[0] == indices0[0]) & (all_matches_origin[0] == -1)).sum() / (all_matches_origin[0] == -1).sum()).item()
                    if (all_matches_origin[0] == -1).sum() > 0
                    else None
                ),
            }
        else:
            return {
                "match_scores": scores[:, :-1, :][0],
                "matches0": indices0[0],  # use -1 for invalid match
                "matching_scores0": mscores0[0],
                "loss": -1,
                "skip_train": True,
                "accuracy": -1,
                "area_accuracy": -1,
                "valid_accuracy": -1,
                "invalid_accuracy": -1,
            }
