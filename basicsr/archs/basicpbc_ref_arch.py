import argparse
import matplotlib.pyplot as plt
import open_clip
import os
import torch
import torch.nn.functional as F
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from random import randint, random
from torch import nn
from torch.nn import init
from torch_scatter import scatter as super_pixel_pooling
from torchvision.transforms.functional import to_pil_image

from basicsr.utils.registry import ARCH_REGISTRY
from raft.raft import RAFT

all_text_labels=['background', 'bag', 'belt', 'glasses', 'hair', 'socks', 'hat', 'mouth', 'clothes', 'eye', 'shoes', 'skin']
def MLP(channels: list, do_bn=True):
    """Multi-layer perceptron"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                # layers.append(nn.BatchNorm1d(channels[i]))
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


def token2img(token):
    line, mask = token[0], token[1]
    h, w = line.shape
    img = torch.ones(3, h, w)
    img[:, line < 1.0] = 0.0
    img[0, mask > 0.0] = 0.9
    img[1, mask > 0.0] = 0.9
    img[2, mask > 0.0] = 0.0
    return img


def save_images(img_list, save_dir):
    for i, img in enumerate(img_list):
        np_img = img.permute(1, 2, 0).numpy()
        filename = f"image_{i}.png"
        image_path = os.path.join(save_dir, filename)
        plt.imsave(image_path, np_img)
    return


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                    # nn.InstanceNorm2d(reduction_dim),
                    nn.PReLU(),
                )
            )
        self.features = nn.ModuleList(self.features)
        self.fuse = nn.Sequential(nn.Conv2d(in_dim + reduction_dim * 4, in_dim, kernel_size=3, padding=1, bias=False), nn.PReLU())

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode="bilinear", align_corners=True))
        out_feat = self.fuse(torch.cat(out, 1))
        return out_feat


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


def get_images_from_tokens(tokens, save=False, save_path="visualization"):

    # tokens: a stack of token. n*c*h*w
    imgs = [token2img(token) for token in tokens.squeeze()]

    if save:
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join(save_path, timestamp)
        os.makedirs(save_dir, exist_ok=True)
        save_images(imgs, save_dir)

    return imgs


class UNet(nn.Module):
    def __init__(self, enc_dim, ch_in=3, use_ppm=False, use_clip=False, bins=[1, 2, 3, 6]):
        super(UNet, self).__init__()

        self.use_ppm = use_ppm
        self.use_clip = use_clip

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=ch_in, ch_out=16)
        self.Conv2 = conv_block(ch_in=16, ch_out=32)
        self.Conv3 = conv_block(ch_in=32, ch_out=64)
        self.Conv4 = conv_block(ch_in=64, ch_out=128)
        self.Conv5 = conv_block(ch_in=128, ch_out=256)

        if use_ppm:
            self.ppm = PPM(496, 128, bins=bins)

        self.Up5 = up_conv(ch_in=496, ch_out=256)
        self.Up_conv5 = conv_block(ch_in=496, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv4 = conv_block(ch_in=240, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv3 = conv_block(ch_in=112, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv2 = conv_block(ch_in=48, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32, enc_dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.Sequential(nn.InstanceNorm2d(enc_dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        init_type = "normal"
        gain = 0.02
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        # encoding path
        x_resized1 = F.avg_pool2d(x, kernel_size=2, stride=2)  # H/2, W/2
        x_resized2 = F.avg_pool2d(x_resized1, kernel_size=2, stride=2)  # H/4, W/4
        x_resized3 = F.avg_pool2d(x_resized2, kernel_size=2, stride=2)  # H/8, W/8
        x_resized4 = F.avg_pool2d(x_resized3, kernel_size=2, stride=2)  # H/16, W/16

        # orignal input
        x1 = self.Conv1(x)  # 16, H, W

        x2 = self.Maxpool(x1)  # H/2, W/2
        x2 = self.Conv2(x2)  # 32, H/2, W/2

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)  # 64, H/4, W/4

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)  # 128, H/8, W/8

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)  # 256, H/16, W/16

        # downsample 1
        x1_resized1 = self.Conv1(x_resized1)  # 16, H/2, W/2

        x2_resized1 = self.Maxpool(x1_resized1)
        x2_resized1 = self.Conv2(x2_resized1)  # 32, H/4, W/4

        x3_resized1 = self.Maxpool(x2_resized1)
        x3_resized1 = self.Conv3(x3_resized1)  # 64, H/8, W/8

        x4_resized1 = self.Maxpool(x3_resized1)
        x4_resized1 = self.Conv4(x4_resized1)  # 128, H/16, W/16

        # downsample 2
        x1_resized2 = self.Conv1(x_resized2)  # 16, H/4, W/4

        x2_resized2 = self.Maxpool(x1_resized2)
        x2_resized2 = self.Conv2(x2_resized2)  # 32, H/8, W/8

        x3_resized2 = self.Maxpool(x2_resized2)
        x3_resized2 = self.Conv3(x3_resized2)  # 64, H/16, W/16

        # downsample 3
        x1_resized3 = self.Conv1(x_resized3)  # 16, H/8, W/8

        x2_resized3 = self.Maxpool(x1_resized3)
        x2_resized3 = self.Conv2(x2_resized3)  # 32, H/16, W/16

        # downsample 4
        x1_resized4 = self.Conv1(x_resized4)  # 16, H/16, W/16

        x1 = x1  # 16 ,H, W
        x2 = torch.cat((x2, x1_resized1), dim=1)  # 48 (32+16) ,H/2, H/2
        x3 = torch.cat((x3, x2_resized1, x1_resized2), dim=1)  # 112 (64+32+16) ,H/4, H/4
        x4 = torch.cat((x4, x3_resized1, x2_resized2, x1_resized3), dim=1)  # 240 (128+64+32+16) ,H/8, H/8
        x5 = torch.cat((x5, x4_resized1, x3_resized2, x2_resized3, x1_resized4), dim=1)  # 496 (256+128+64+32+16) ,H/16, H/16

        if self.use_ppm:
            x5 = self.ppm(x5)  # 512, H/16, W/16

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)  # 240+256
        d5 = self.Up_conv5(d5)  # 496->256

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)  # 112+128
        d4 = self.Up_conv4(d4)  # 240->128

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)  # 48+64
        d3 = self.Up_conv3(d3)  # 240->128

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)  # 16+32
        d2 = self.Up_conv2(d2)  # 48->32

        d1 = self.Conv_1x1(d2)  # 32->3
        d1 = self.norm(d1)
        return d1


class SegmentDescriptor(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, enc_dim, ch_in=3, use_ppm=False, use_clip=False):
        super().__init__()
        self.encoder = UNet(enc_dim, ch_in, use_ppm, use_clip)
        # self.super_pixel_pooling =
        # use scatter
        # nn.init.constant_(self.encoder[-1].bias, 0.0)

    def extract_feat(self, img):
        img_resized = F.interpolate(img, size=(640, 640), mode="bilinear", align_corners=False)
        x_resized = self.encoder(img_resized)
        x = F.interpolate(x_resized, size=(img.shape[2], img.shape[3]), mode="bilinear", align_corners=False)
        return x

    def tokenization(self, x, seg):
        n, c, h, w = x.size()
        return super_pixel_pooling(x.view(n, c, -1), seg.view(-1).long(), reduce="mean")

    def forward(self, img, seg):
        img_resized = F.interpolate(img, size=(640, 640), mode="bilinear", align_corners=False)
        x_resized = self.encoder(img_resized)
        x = F.interpolate(x_resized, size=(img.shape[2], img.shape[3]), mode="bilinear", align_corners=False)
        n, c, h, w = x.size()
        assert (h, w) == img.size()[2:4]
        return super_pixel_pooling(x.view(n, c, -1), seg.view(-1).long(), reduce="mean")
        # here return size is [1]xCx|Seg|


class KeypointEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([4] + layers + [feature_dim])
        # for m in self.encoder.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        inputs = kpts.transpose(1, 2)
        # print(inputs.size(), 'wula!')
        x = self.encoder(inputs)
        # print(x.size())
        return x

'''
class TextClipEncoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        self.clip, _, _ = open_clip.create_model_and_transforms("convnext_large_d_320", pretrained="laion2b_s29b_b131k_ft_soup") 
        self.tokenizer = open_clip.get_tokenizer("convnext_large_d_320")
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = [t[0] for t in x]
        x = self.tokenizer(x).cuda()
        x = self.clip.encode_text(x)
        x /= x.norm(dim=-1, keepdim=True)  # s, d
        return x


class TagFeatureEncoder(nn.Module):
    def __init__(self, enc_dim):
        super().__init__()
        self.clip = TextClipEncoder()
        self.fc = nn.Sequential(nn.Linear(768, enc_dim), nn.ReLU(), nn.Linear(enc_dim, enc_dim))

    def forward(self, x):
        x = self.clip(x)  # s, d
        x = self.fc(x).unsqueeze(0).permute(0, 2, 1)  # n, d, s
        return x
'''

class TagFeatureEncoder(nn.Module):
    def __init__(self, enc_dim):
        super().__init__()
        self.fc = nn.Linear(12, enc_dim)
        text_linear_path="/data/dyk/Segment_Matching/BasicPBC_release/BasicPBC/ckpt/text_linear.pth"
        loaded_model = torch.load(text_linear_path)
        self.fc.load_state_dict(loaded_model.state_dict())

    def forward(self, x):
        indices = [all_text_labels.index(tag[0]) for tag in x]
        one_hot = torch.nn.functional.one_hot(torch.tensor(indices), num_classes=len(all_text_labels)).float().unsqueeze(0).to(self.fc.weight.device) #B,N,12
        x = self.fc(one_hot) #B,N,128
        return x.permute(0,2,1) # B 128 N


class Fuse(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Linear(dim_out, dim_out))

    def forward(self, x):
        x = self.mlp(x)
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


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class GumbelQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, num_hiddens, straight_through=False, kl_weight=5e-4, temp_init=1.0):
        super().__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight
        # self.proj = nn.Conv2d(num_hiddens, codebook_size, 1)  # projects last encoder layer to quantized logits
        # self.embed = nn.Embedding(codebook_size, emb_dim) #codebook

    def forward(self, z, word_codebook):
        hard = self.straight_through if self.training else True

        logits = self.proj(z)  # bchw -> bnhw, n is codebook size

        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)

        z_q = torch.einsum("b n h w, n d -> b d h w", soft_one_hot, self.embed.weight)

        qy = F.softmax(logits, dim=1)

        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.codebook_size + 1e-10), dim=1).mean()  # codebook loss

        min_encoding_indices = soft_one_hot.argmax(dim=1)

        return z_q, diff, {"min_encoding_indices": min_encoding_indices}


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


@ARCH_REGISTRY.register()
class BasicPBC_ref(nn.Module):
    """Main architecture for the PBC model ref"""

    def __init__(
        self, descriptor_dim=128, keypoint_encoder=[32, 64, 128], GNN_layer_num=9, text_loss_weight=0.0, wo_text=False, wo_parsing=False, use_raft=False
    ):

        super().__init__()

        config = argparse.Namespace()
        config.descriptor_dim = descriptor_dim
        config.keypoint_encoder = keypoint_encoder
        config.GNN_layers_num = GNN_layer_num
        config.GNN_layers = ["self", "cross"] * GNN_layer_num
        config.text_loss_weight = text_loss_weight

        config.wo_text = wo_text
        config.wo_parsing = wo_parsing
        config.use_raft = use_raft
        config.raft_resolution = (320, 320)

        if use_raft:
            config.ch_in = 6
        else:
            config.ch_in = 3

        if wo_parsing:
            config.ch_in_ref = 3
        else:
            config.ch_in_ref = 6

        # self.config = {**self.default_config, **config}

        self.config = config

        bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter("bin_score", bin_score)

        # stage1
        self.desc_line = SegmentDescriptor(self.config.descriptor_dim, self.config.ch_in, use_ppm=True, use_clip=False)
        self.desc_parse = SegmentDescriptor(self.config.descriptor_dim, self.config.ch_in_ref, use_ppm=True, use_clip=False)

        if self.config.wo_parsing and self.config.wo_text:
            pass
        else:
            self.fuse_tar = Fuse(2 * self.config.descriptor_dim, self.config.descriptor_dim)
            if self.config.wo_text:
                self.fuse_ref = Fuse(self.config.descriptor_dim, self.config.descriptor_dim)
            else:
                self.fuse_ref = Fuse(2 * self.config.descriptor_dim, self.config.descriptor_dim)

        # self.clip_text = TextFeatureEncoder(self.config.descriptor_dim)
        self.clip_text = TagFeatureEncoder(self.config.descriptor_dim)

        self.kenc = KeypointEncoder(self.config.descriptor_dim, self.config.keypoint_encoder)

        self.gnn = AttentionalGNN(self.config.descriptor_dim, self.config.GNN_layers)
        self.final_proj = nn.Conv1d(self.config.descriptor_dim, self.config.descriptor_dim, kernel_size=1, bias=True)

        if self.config.use_raft:
            args = {
                "mixed_precision": False,
                "small": False,
                "ckpt": "raft/ckpt/raft-animerun-v2-ft_again.pth",
                "freeze": True,
            }

            self.raft = RAFT(args)
            state_dict = torch.load(args["ckpt"])
            real_state_dict = {k.split("module.")[-1]: v for k, v in state_dict.items()}
            self.raft.load_state_dict(real_state_dict)
            for param in self.raft.parameters():
                param.requires_grad = False

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""

        # segment parsing module

        #  -----  Line Feature Extraction  ------
        # tar
        input_tar = data["line"]

        if not self.config.wo_parsing:
            input_parse_tar = torch.cat([input_tar, data["parse_mask"]], dim=1)
            seq_parse_tar = self.desc_parse(input_parse_tar, data["segment"])[..., 1:]  # 1, d, n
        else:
            seq_parse_tar = self.desc_parse(input_tar, data["segment"])[..., 1:]  # 1, d, n

        # ref
        input_ref = data["line_refs"]
        seq_tag_ref = self.clip_text(data["used_tags_ref"])  # 1, d, m
        #print("seg_tag", seq_tag_ref, seq_tag_ref.shape)
        seg_tag_indices = [data["used_tags_ref"].index(tag) for tag in data["seg_tags_refs"]]  # indices of the segment tags

        if self.config.use_raft:
            h, w = input_tar.shape[-2:]
            color_ref = data["colored_gt_refs"]
            line_tar = F.interpolate(input_tar, self.config.raft_resolution, mode="bilinear", align_corners=False)
            line_ref = F.interpolate(input_ref, self.config.raft_resolution, mode="bilinear", align_corners=False)
            color_ref_resize = F.interpolate(color_ref, self.config.raft_resolution, mode="bilinear", align_corners=False)

            self.raft.eval()
            _, flow_up = self.raft(line_tar, line_ref, iters=20, test_mode=True)
            color_tar = flow_warp(color_ref_resize, flow_up.permute(0, 2, 3, 1).detach(), "nearest")
            color_tar = F.interpolate(color_tar, (h, w), mode="bilinear", align_corners=False)

        #  -----  Segment Parsing  -------

        # temp_init = 1.0
        # hard = False if self.training else True
        logits = torch.einsum("b d n, b d m -> b n m", seq_parse_tar, seq_tag_ref)
        # soft_one_hot = F.gumbel_softmax(logits, tau=temp_init, dim=2, hard=hard)
        # feat_tag_match = torch.einsum("b n m, b d m -> b d n", soft_one_hot, feat_tag_ref)
        # feat_fuse = self.fuse_text(torch.cat([feat_stage1, feat_tag_match], dim=1).permute(0, 2, 1)).permute(0, 2, 1)

        text_scores = transport(logits / self.config.descriptor_dim**0.5, self.bin_score)[:, :-1, :].squeeze(0)  # n, m+1
        text_scores = nn.functional.softmax(text_scores, dim=-1)
        _, pred_indices = torch.max(text_scores, 1)
        mapping_list = [tag[0] for tag in data["used_tags_ref"]]
        mapping_list.append("background")  # for unmatched region
        pred_text_labels = [mapping_list[i] for i in pred_indices]

        tag_all_matches = data["tag_all_matches"] if "tag_all_matches" in data else None
        if tag_all_matches is not None and self.config.text_loss_weight > 0:
            tag_all_matches[tag_all_matches == -1] = logits.shape[-1]
            text_loss = nn.functional.cross_entropy(text_scores, tag_all_matches.long().view(-1), reduction="mean")

        #  -----  Segment Matching  -------
        if self.config.use_raft:
            input_tar = torch.cat((input_tar, color_tar), dim=1)
            input_ref = torch.cat((input_ref, color_ref), dim=1)
        seq_tar = self.desc_line(input_tar, data["segment"])[..., 1:]  # 1, d, n

        if self.config.wo_parsing and self.config.wo_text:
            seq_fuse_tar = seq_tar

            seq_segment_ref = self.desc_line(input_ref, data["segment_refs"])[..., 1:]  # 1, d, m (num of ref segments)
            seq_segment_fuse_ref = seq_segment_ref
        else:
            # tar
            seq_cat_tar = torch.cat([seq_tar, seq_parse_tar], dim=1)
            seq_fuse_tar = self.fuse_tar(seq_cat_tar.permute(0, 2, 1)).permute(0, 2, 1)

            # ref
            seq_segment_ref = self.desc_line(input_ref, data["segment_refs"])[..., 1:]  # 1, d, m (num of ref segments)

            if self.config.wo_text:
                seq_segment_fuse_ref = self.fuse_ref(seq_segment_ref.permute(0, 2, 1)).permute(0, 2, 1)  # 1, d, m
            else:
                seq_segment_cat_ref = torch.cat([seq_segment_ref, seq_tag_ref[:, :, seg_tag_indices]], dim=1)
                seq_segment_fuse_ref = self.fuse_ref(seq_segment_cat_ref.permute(0, 2, 1)).permute(0, 2, 1)  # 1, d, m

        kpts, kpts_ref = data["keypoints"].float(), data["keypoints_refs"].float()

        if kpts.shape[1] < 2 or kpts_ref.shape[1] < 2:  # no keypoints
            shape0, shape1 = kpts.shape[:-1], kpts_ref.shape[:-1]
            print(data["file_name"])
            return {"matches0": kpts.new_full(shape0, -1, dtype=torch.int)[0], "matching_scores0": kpts.new_zeros(shape0)[0], "skip_train": True}

        kpts = normalize_keypoints(kpts, data["line"].shape)
        kpts_ref = normalize_keypoints(kpts_ref, data["line_refs"].shape)

        pos = self.kenc(kpts)
        pos_ref = self.kenc(kpts_ref)

        feat_final = seq_fuse_tar + pos
        feat_final_ref = seq_segment_fuse_ref + pos_ref

        # feat_final = self.fuse_tar(torch.cat([feat_stage2, feat_tag_match], dim=1).permute(0, 2, 1)).permute(0, 2, 1)
        # feat_final_ref = self.fuse_ref(torch.cat([feat_stage2_ref, feat_tag_ref[:, :, seg_tag_indices]], dim=1).permute(0, 2, 1)).permute(0, 2, 1)

        # Multi-layer Transformer network.
        feat_final, feat_final_ref = self.gnn(feat_final, feat_final_ref)

        # Final MLP projection.
        mdesc, mdesc_ref = self.final_proj(feat_final), self.final_proj(feat_final_ref)

        # Compute matching descriptor distance.
        scores = torch.einsum("bdn,bdm->bnm", mdesc, mdesc_ref)
        scores /= self.config.descriptor_dim**0.5

        # Run the optimal transport.
        b, m, n = scores.size()
        scores = transport(scores, self.bin_score)  # 1, n+1, m+1

        seg_all_matches = data["seg_all_matches"] if "seg_all_matches" in data else None
        all_matches_origin = seg_all_matches.clone() if seg_all_matches is not None else None
        weights = data["numpts"].float()  # .cuda()

        if seg_all_matches is not None:
            seg_all_matches[seg_all_matches == -1] = n
            loss = nn.functional.cross_entropy(scores[:, :-1, :].view(-1, n + 1), seg_all_matches.long().view(-1), reduction="mean")
            # loss = loss.mean()

            if self.config.text_loss_weight > 0:
                alpha = self.config.text_loss_weight
                loss += alpha * text_loss

        scores = nn.functional.softmax(scores, dim=2)

        max0, max1 = scores[:, :-1, :].max(2), scores[:, :, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mscores0 = max0.values

        valid0 = indices0 < n
        valid1 = indices1 < m
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        if seg_all_matches is not None:
            return {
                "match_scores": scores[:, :-1, :][0],
                "text_scores": text_scores if self.config.text_loss_weight > 0 else None,
                "pred_labels": pred_text_labels if self.config.text_loss_weight > 0 else None,
                "matches0": indices0[0],  # use -1 for invalid match
                "matching_scores0": mscores0[0],
                "loss": loss,
                "skip_train": False,
                "accuracy": ((all_matches_origin[0] == indices0[0]).sum() / len(all_matches_origin[0])).item(),
                "area_accuracy": (
                    torch.tensor(
                        [
                            (data["segment"] == ii + 1).sum()
                            for ii in torch.arange(0, all_matches_origin[0].shape[0]).to(data["segment"])[all_matches_origin[0] == indices0[0]]
                        ]
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
                "text_scores": text_scores if self.config.text_loss_weight > 0 else None,
                "pred_labels": pred_text_labels if self.config.text_loss_weight > 0 else None,
                "matches0": indices0[0],  # use -1 for invalid match
                "matching_scores0": mscores0[0],
                "loss": -1,
                "skip_train": True,
                "accuracy": -1,
                "area_accuracy": -1,
                "valid_accuracy": -1,
            }
