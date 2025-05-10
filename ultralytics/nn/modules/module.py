import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from .conv import Conv, RepConv
from .head import Detect
from einops import rearrange
from .module_v2 import (CSMHSA, SelfAttention, DynamicResidualBlock, ShuffledGroupConv)
from ultralytics.nn.modules.block import (C2f, DFL)
from ultralytics.nn.modules.conv import autopad
from ultralytics.yolo.utils.tal import (dist2bbox, make_anchors)

__all__ = [
    'DySample', 'CARAFE', 'SPDConv', 'CSPOmniKernel']

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


class CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x

######################################## Omni-Kernel Network for Image Restoration [AAAI-24] start ########################################

class FGM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1, groups=dim)

        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        # res = x.clone()
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='backward')

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta

class OmniKernel(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 31 # 7
        pad = ker // 2
        self.in_conv = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
                    nn.GELU()
                    )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,pad), stride=1, groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(pad,0), stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.ReLU()

        ### sca ###
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        ### fca ###
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)

        ### fca ###
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2,-1), norm='backward')
        x_fca = torch.abs(x_fca)

        ### fca ###
        ### sca ###
        x_att = self.conv(self.pool(x_fca))
        x_sca = x_att * x_fca
        ### sca ###
        x_sca = self.fgm(x_sca)

        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)

class CSPOmniKernel(nn.Module):
    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = OmniKernel(int(dim * self.e))

    def forward(self, x):
        ok_branch, identity = torch.split(self.cv1(x), [int(self.cv1.conv.out_channels * self.e), int(self.cv1.conv.out_channels * (1 - self.e))], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))

######################################## Omni-Kernel Network for Image Restoration [AAAI-24] end ########################################

################################################### LXD
class DynamicInceptionDWConv2d(nn.Module):
    """ Dynamic Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        self.dwconv = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, square_kernel_size, padding=square_kernel_size // 2,
                      groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                      groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                      groups=in_channels)
        ])

        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()

        # Dynamic Kernel Weights
        self.dkw = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels * 3, 1)
        )

    def forward(self, x):
        x_dkw = rearrange(self.dkw(x), 'bs (g ch) h w -> g bs ch h w', g=3)
        x_dkw = F.softmax(x_dkw, dim=0)
        x = torch.stack([self.dwconv[i](x) * x_dkw[i] for i in range(len(self.dwconv))]).sum(0)
        return self.act(self.bn(x))

class DICSPOmniKernel(nn.Module):
    def __init__(self, dim, e1=0.25, e2=0.25, kernels=[1, 3]):
        super().__init__()
        self.e1 = e1
        self.e2 = e2
        self.e3 = 1.0-(e1+e2)
        min_ch = dim//4
        self.cv = Conv(dim, dim, 1)
        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(DynamicInceptionDWConv2d(min_ch, ks, ks * 3 + 2))

        self.m = OmniKernel(int(dim * self.e3))

    def forward(self, x):
        group_input = torch.split(x,  [int(x.size(1) * self.e1), int(x.size(1) * self.e2), int(x.size(1) * self.e3)], dim=1)
        out1 = self.convs[0](group_input[0])
        out2 = self.convs[1](group_input[1])
        out3 = self.m(group_input[2])
        return self.cv(torch.cat([out1, out2, out3], dim=1))


class DICSPOmniKernelv2(nn.Module):
    def __init__(self, dim, e1=0.25, e2=0.25, kernels=[1, 3]):
        super().__init__()
        self.e1 = e1
        self.e2 = e2
        self.e3 = 1.0-(e1+e2)
        min_ch = dim//4
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(int(dim * self.e3), int(dim * self.e3)//2, 1)
        self.cv3 = Conv(int(dim * self.e3)//2, int(dim * self.e3), 1)
        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(DynamicInceptionDWConv2d(min_ch, ks, ks * 3 + 2))

        self.m = OmniKernel(int(dim * self.e3)//2)

    def forward(self, x):
        group_input = torch.split(x,  [int(x.size(1) * self.e1), int(x.size(1) * self.e2), int(x.size(1) * self.e3)], dim=1)
        out1 = self.convs[0](group_input[0])
        out2 = self.convs[1](group_input[1])

        out3 =self.cv3(self.m(self.cv2(group_input[2])))
        return self.cv1(torch.cat([out1, out2, out3], dim=1))

class DICSPv2_Residual(nn.Module):
    def __init__(self, dim, e1=0.25, e2=0.25, kernels=[1, 3]):
        super().__init__()
        self.e1 = e1
        self.e2 = e2
        self.e3 = 1.0-(e1+e2)
        min_ch = dim//4
        self.cv1 = Conv(int(1.5 * dim), dim, 1)
        self.cv2 = Conv(int(dim * self.e3), int(dim * self.e3)//2, 1)
        self.cv3 = Conv(int(dim * self.e3)//2, int(dim * self.e3), 1)
        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(DynamicInceptionDWConv2d(min_ch, ks, ks * 3+ 2))

        self.m = nn.ModuleList(GatedCNNBlock_BCHW(int(dim * self.e3)//2) for _ in range(1))
        # self.m = OmniKernel(int(dim * self.e3)//2)

    def forward(self, x):
        group_input = torch.split(x,  [int(x.size(1) * self.e1), int(x.size(1) * self.e2), int(x.size(1) * self.e3)], dim=1)
        out1 = self.convs[0](group_input[0])
        out2 = self.convs[1](group_input[1])

        out3 =self.cv3(self.m[0](self.cv2(group_input[2]))) # self.m
        return self.cv1(torch.cat([out1, out2, out3, group_input[2]], dim=1))

class DICSPv2_Residual_v2(nn.Module):
    # 添加 layer_level 参数，默认为 'deep'
    def __init__(self, dim, e1=0.25, e2=0.25, kernels=[1, 3], layer_level='deep'):
        super().__init__()
        self.e1 = e1
        self.e2 = e2
        self.e3 = 1.0 - (e1 + e2)
        self.layer_level = layer_level

        channels_e1 = int(dim * self.e1)
        channels_e2 = int(dim * self.e2)
        channels_e3 = dim - channels_e1 - channels_e2
        self.split_dims = [channels_e1, channels_e2, channels_e3]

        self.convs = nn.ModuleList()
        self.convs.append(DynamicInceptionDWConv2d(channels_e1, kernels[0], kernels[0] * 3 + 2))

        # 分支2 (对应 group_input[1]) 的卷积只在 'deep' 层级需要
        if self.layer_level == 'deep':
            self.convs.append(DynamicInceptionDWConv2d(channels_e2, kernels[1], kernels[1] * 3 + 2))
        # --- 分支3 (对应 group_input[2]) 的模块总是需要的 ---
        if channels_e3 > 0:
            hidden_channels_e3 = channels_e3 // 2
        self.cv2 = Conv(channels_e3, hidden_channels_e3, 1)
        self.m = nn.ModuleList(GatedCNNBlock_BCHW(hidden_channels_e3) for _ in range(1))
        self.cv3 = Conv(hidden_channels_e3, channels_e3, 1)

        # --- 定义最终的卷积层 self.cv1 ---
        concat_channels = channels_e1 + channels_e2 + channels_e3 + channels_e3
        self.cv1 = Conv(concat_channels, dim, 1)


    def forward(self, x):

        group_input = torch.split(x, self.split_dims, dim=1)
        out1 = self.convs[0](group_input[0]) if self.split_dims[0] > 0 else group_input[0]
        out3 = self.cv3(self.m[0](self.cv2(group_input[2])))

        if self.layer_level == 'shallow':
            concat_list = [out1, group_input[1], out3, group_input[2]]
        else: # deep
            out2 = self.convs[1](group_input[1]) if self.split_dims[1] > 0 else group_input[1]
            concat_list = [out1, out2, out3, group_input[2]]

        return self.cv1(torch.cat(concat_list, dim=1))

class DICSPv2_Residual_add_v2(nn.Module):
    # 添加 layer_level 参数，默认为 'deep'
    def __init__(self, dim, kernels=[1, 3], layer_level='deep'):
        super().__init__()
        self.e1 = 0.25
        self.e2 = 0.25
        self.e3 = 1.0 - (self.e1 + self.e2)
        self.layer_level = layer_level

        channels_e1 = int(dim * self.e1)
        channels_e2 = int(dim * self.e2)
        channels_e3 = dim - channels_e1 - channels_e2
        self.split_dims = [channels_e1, channels_e2, channels_e3]

        self.convs = nn.ModuleList()
        self.convs.append(DynamicInceptionDWConv2d(channels_e1, kernels[0], kernels[0] * 3 + 2))

        # 分支2 (对应 group_input[1]) 的卷积只在 'deep' 层级需要
        if self.layer_level == 'deep':
            self.convs.append(DynamicInceptionDWConv2d(channels_e2, kernels[1], kernels[1] * 3 + 2))
        # --- 分支3 (对应 group_input[2]) 的模块总是需要的 ---
        if channels_e3 > 0:
            hidden_channels_e3 = channels_e3 // 2
        self.cv2 = Conv(channels_e3, hidden_channels_e3, 1)
        self.m = nn.ModuleList(GatedCNNBlock_BCHW(hidden_channels_e3) for _ in range(1))
        self.cv3 = Conv(hidden_channels_e3, channels_e3, 1)

        # --- 定义最终的卷积层 self.cv1 ---
        concat_channels = channels_e1 + channels_e2 + channels_e3 + channels_e3
        self.cv1 = Conv(concat_channels, dim, 1)


    def forward(self, x):

        group_input = torch.split(x, self.split_dims, dim=1)
        out1 = self.convs[0](group_input[0]) + group_input[0]
        out3 = self.cv3(self.m[0](self.cv2(group_input[2])))

        if self.layer_level == 'shallow':
            concat_list = [out1, group_input[1], out3, group_input[2]]
        else: # deep
            out2 = self.convs[1](group_input[1]) + group_input[1]
            concat_list = [out1, out2, out3, group_input[2]]

        return self.cv1(torch.cat(concat_list, dim=1))

class DICSPOmniKernelv3(nn.Module):
    def __init__(self, dim, e1=0.75, e2=0.25, kernels=[1, 3]):
        super().__init__()
        self.e1 = e1
        self.e2 = e2
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(int(dim * self.e2), int(dim * self.e2)//2, 1)
        self.cv3 = Conv(int(dim * self.e2)//2, int(dim * self.e2), 1)
        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(DynamicInceptionDWConv2d(int(dim * self.e1), ks, ks * 2 + 1)) # ks * 3 + 2

        self.m = OmniKernel(int(dim * self.e2)//2)

    def forward(self, x):
        group_input = torch.split(x,  [int(x.size(1) * self.e1), int(x.size(1) * self.e2)], dim=1)
        out1 = self.convs[0](group_input[0])
        out2 = self.convs[1](group_input[0])
        out3 = group_input[0] + out1 + out2

        out4 =self.cv3(self.m(self.cv2(group_input[1])))
        return self.cv1(torch.cat([out3, out4], dim=1))

class DICSPOmniKernelv4(nn.Module):
    def __init__(self, dim, e1=0.75, e2=0.25, kernels=[1, 3]):
        super().__init__()
        self.e1 = e1
        self.e2 = e2
        self.cv1 = Conv(int(dim + self.e1 * dim), dim, 1)
        self.cv2 = Conv(int(dim * self.e2), int(dim * self.e2)//2, 1)
        self.cv3 = Conv(int(dim * self.e2)//2, int(dim * self.e2), 1)
        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(DynamicInceptionDWConv2d(int(dim * self.e1), ks, ks * 2 + 1)) # ks * 3 + 2

        self.m = OmniKernel(int(dim * self.e2)//2)

    def forward(self, x):
        group_input = torch.split(x,  [int(x.size(1) * self.e1), int(x.size(1) * self.e2)], dim=1)
        out1 = self.convs[0](group_input[0])
        out2 = self.convs[1](group_input[0])
        out3 = out1 + out2

        out4 =self.cv3(self.m(self.cv2(group_input[1])))
        return self.cv1(torch.cat([out3, out4, group_input[0]], dim=1))
##################################################################

##################################################################
######################################## CVPR2025 MambaOut start ########################################
class LayerNormGeneral(nn.Module):
    r""" General LayerNorm for different situations.

    Args:
        affine_shape (int, list or tuple): The shape of affine weight and bias.
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm,
            the affine_shape is the same as normalized_dim by default.
            To adapt to different situations, we offer this argument here.
        normalized_dim (tuple or list): Which dims to compute mean and variance.
        scale (bool): Flag indicates whether to use scale or not.
        bias (bool): Flag indicates whether to use scale or not.

        We give several examples to show how to specify the arguments.

        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.

        Modified LayerNorm (https://arxiv.org/abs/2111.11418)
            that is idental to partial(torch.nn.GroupNorm, num_groups=1):
            For input shape of (B, N, C),
                affine_shape=C, normalized_dim=(1, 2), scale=True, bias=True;
            For input shape of (B, H, W, C),
                affine_shape=C, normalized_dim=(1, 2, 3), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, 2, 3), scale=True, bias=True.

        For the several metaformer baslines,
            IdentityFormer, RandFormer and PoolFormerV2 utilize Modified LayerNorm without bias (bias=False);
            ConvFormer and CAFormer utilizes LayerNorm without bias (bias=False).
    """
    def __init__(self, affine_shape=None, normalized_dim=(-1, ), scale=True,
        bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x
class GatedCNNBlock_BCHW(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args:
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve practical efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(LayerNormGeneral,eps=1e-6,normalized_dim=(1, 2, 3)),
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer((dim, 1, 1))
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        # c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        # c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=1))
        x = self.drop_path(x)
        return x + shortcut

class C2f_MambaOut(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(GatedCNNBlock_BCHW(self.c) for _ in range(n))

######################################## CVPR2025 MambaOut end ########################################
##################################################################

######################################## DynamicConvMixerBlock start ########################################

class DynamicInceptionDWConv2d(nn.Module):
    """ Dynamic Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        self.dwconv = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, square_kernel_size, padding=square_kernel_size // 2,
                      groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                      groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                      groups=in_channels)
        ])

        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()

        # Dynamic Kernel Weights
        self.dkw = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels * 3, 1)
        )

    def forward(self, x):
        x_dkw = rearrange(self.dkw(x), 'bs (g ch) h w -> g bs ch h w', g=3)
        x_dkw = F.softmax(x_dkw, dim=0)
        x = torch.stack([self.dwconv[i](x) * x_dkw[i] for i in range(len(self.dwconv))]).sum(0)
        return self.act(self.bn(x))


class DynamicInceptionMixer(nn.Module):
    def __init__(self, channel=256, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // 2

        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(DynamicInceptionDWConv2d(min_ch, ks, ks * 3 + 2))
        self.conv_1x1 = Conv(channel, channel, k=1)

    def forward(self, x):
        _, c, _, _ = x.size()
        x_group = torch.split(x, [c // 2, c // 2], dim=1)
        x_group = torch.cat([self.convs[i](x_group[i]) for i in range(len(self.convs))], dim=1)
        x = self.conv_1x1(x_group)
        return x


class DynamicIncMixerBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mixer = DynamicInceptionMixer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = ConvolutionalGLU(dim)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class C2f_DCMB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(DynamicIncMixerBlock(self.c) for _ in range(n))


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                      groups=hidden_features),
            act_layer()
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    # def forward(self, x):
    #     x, v = self.fc1(x).chunk(2, dim=1)
    #     x = self.dwconv(x) * v
    #     x = self.drop(x)
    #     x = self.fc2(x)
    #     x = self.drop(x)
    #     return x

    def forward(self, x):
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

######################################## DynamicConvMixerBlock end ########################################

####################################### Detect_LSCD #######################################################
class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

class Conv_GN(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.gn(self.conv(x)))

class Detect_LSCSBD(nn.Module):
    # Lightweight Shared Convolutional Separate BN Detection Head
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, hidc=256, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.conv = nn.ModuleList(nn.Sequential(Conv(x, hidc, 1)) for x in ch)
        self.share_conv = nn.Sequential(nn.Conv2d(hidc, hidc, 3, 1, 1), nn.Conv2d(hidc, hidc, 3, 1, 1))
        self.separate_bn = nn.ModuleList(nn.Sequential(nn.BatchNorm2d(hidc), nn.BatchNorm2d(hidc)) for _ in ch)
        self.act = nn.SiLU()
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
            for j in range(len(self.share_conv)):
                x[i] = self.act(self.separate_bn[j](self.share_conv[j](x[i])))
            x[i] = torch.cat((self.scale[i](self.cv2(x[i])), self.cv3(x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        # for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
        m.cv2.bias.data[:] = 1.0  # box
        m.cv3.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes):
        """Decode bounding boxes."""
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
########################################################################################################################

############################################## LXD segment_v2 ##########################################################
class Segment_v2(Detect):
    """YOLOv8 Segment head for segmentation models."""

    #####LXD
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        self.npr = 32  # intermediate convolutional feature dimension
        self.cv1 = Conv(ch[0], self.npr, k=3)
        self.upsample1 = nn.ConvTranspose2d(self.npr, self.npr // 2, 2, 2, 0, bias=True)
        self.upsample2 = nn.ConvTranspose2d(self.npr//2, self.npr // 2, 2, 2, 0, bias=True)
        self.cv2 = Conv(self.npr//2, self.npr//4, k=3)
        self.cv3 = Conv(self.npr//4, self.nc+1) ###### self.nc+1 means add the background

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.cv3(self.cv2(self.upsample2(self.upsample1(self.cv1(x[0]))))) # mask protos
        if self.training:
            return p
        return p

class Segment_v3(Detect):

    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)  # 假设 Detect 的 __init__ 接受 nc 和 ch

        self.seg_nc = nc
        self.seg_npr = 64 #32
        input_channels = ch[0]

        # 原始的上采样和卷积层 (输出 320x320)
        self.cv1 = Conv(input_channels, self.seg_npr, k=3)  # Input: H/8 x W/8 x C -> H/8 x W/8 x 64
        self.upsample1 = nn.ConvTranspose2d(self.seg_npr, self.seg_npr // 2, 2, 2, 0, bias=True)  # -> H/4 x W/4 x 32
        self.cv2 = Conv(self.seg_npr // 2, self.seg_npr // 2, k=3)  # -> H/4 x W/4 x 32
        self.upsample2 = nn.ConvTranspose2d(self.seg_npr // 2, self.seg_npr // 4, 2, 2, 0,
                                            bias=True)  # -> H/2 x W/2 x 16
        self.cv3 = Conv(self.seg_npr // 4, self.seg_npr // 4, k=3)  # -> H/2 x W/2 x 16
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # -> H x W x 16
        self.cv4 = Conv(self.seg_npr // 4, self.seg_npr // 4, k=3)  # -> H x W x 16
        self.cv_pred = Conv(self.seg_npr // 4, self.seg_nc + 1, k=1)  # -> H x W x (seg_nc+1)

        # Sigmoid 通常在计算损失时与 BCEWithLogitsLoss 结合使用，这里可以不加
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature_map = x[0]  # 通常来自 layer 28
        f1 = self.cv1(feature_map)
        f2 = self.cv2(self.upsample1(f1))
        f3 = self.cv3(self.upsample2(f2))
        f4 = self.cv4(self.upsample3(f3))
        p = self.cv_pred(f4)

        # 如果损失函数不是 BCEWithLogitsLoss，可能需要在这里应用 sigmoid
        # p = self.sigmoid(p)

        # 注意：forward 方法对于训练和推理通常返回相同的东西
        # 在 Trainer 中会根据模式处理输出
        return p

class DGCS(nn.Module):
    # Dynamic Group Convolution Shuffl
    def __init__(self, inc) -> None:
        super().__init__()

        self.c = inc // 2
        self.gconv = Conv(self.c, self.c, g=self.c)
        self.conv1 = Conv(inc, inc, 1)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.c, x.size(1) - self.c], 1)
        x1 = self.gconv(x1)
        # shuffle
        b, n, h, w = x1.size()
        b_n = b * n // 2
        y = x1.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        y = torch.cat((y[0], y[1]), 1)

        x = torch.cat([y, x2], 1)
        return x + self.conv1(x)

class Segment_v4(Detect):

    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)  # 假设 Detect 的 __init__ 接受 nc 和 ch

        self.seg_nc = nc
        self.seg_npr = 64 #32
        inc = ch[0]

        self.DGCS1 = DGCS(inc)  # Input: H/8 x W/8 x C -> H/8 x W/8 x 64
        self.DGCS2 = DGCS(inc//2)
        self.upsample1 = nn.ConvTranspose2d(self.seg_npr, self.seg_npr // 2, 2, 2, 0, bias=True)  # -> H/4 x W/4 x 32
        self.upsample2 = nn.ConvTranspose2d(self.seg_npr // 2, self.seg_npr // 4, 2, 2, 0,
                                            bias=True)  # -> H/2 x W/2 x 16
        self.cv = Conv(self.seg_npr // 4, self.seg_npr // 4, k=3)  # -> H/2 x W/2 x 16
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # -> H x W x 16
        self.cv_pred = Conv(self.seg_npr // 4, self.seg_nc + 1, k=1)  # -> H x W x (seg_nc+1)

        # Sigmoid 通常在计算损失时与 BCEWithLogitsLoss 结合使用，这里可以不加
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature_map = x[0]  # 通常来自 layer 28
        f1 = self.DGCS1(feature_map)
        f2 = self.DGCS2(self.upsample1(f1))
        f3 = self.cv(self.upsample2(f2))
        f4 = self.cv(self.upsample3(f3))
        p = self.cv_pred(f4)

        return p

class Segment_v5(Detect):

    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)  # 假设 Detect 的 __init__ 接受 nc 和 ch

        self.seg_nc = nc
        self.seg_npr = 64 #32
        inc = ch[0]

        self.DGCS1 = DGCS(inc)  # Input: H/8 x W/8 x C -> H/8 x W/8 x 64
        self.DGCS2 = DGCS(inc//2)
        self.DGCS3 = DGCS(self.seg_npr // 4)
        self.upsample1 = nn.ConvTranspose2d(self.seg_npr, self.seg_npr // 2, 2, 2, 0, bias=True)  # -> H/4 x W/4 x 32
        self.upsample2 = nn.ConvTranspose2d(self.seg_npr // 2, self.seg_npr // 4, 2, 2, 0,
                                            bias=True)  # -> H/2 x W/2 x 16
        self.cv = Conv(self.seg_npr // 4, self.seg_npr // 4, k=3)  # -> H/2 x W/2 x 16
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # -> H x W x 16
        self.cv_pred = Conv(self.seg_npr // 4, self.seg_nc + 1, k=1)  # -> H x W x (seg_nc+1)

        # Sigmoid 通常在计算损失时与 BCEWithLogitsLoss 结合使用，这里可以不加
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature_map = x[0]  # 通常来自 layer 28
        f1 = self.DGCS1(feature_map)
        f2 = self.DGCS2(self.upsample1(f1))
        f3 = self.DGCS3(self.upsample2(f2))
        f4 = self.cv(self.upsample3(f3))
        p = self.cv_pred(f4)

        return p

class Segment_v6(Detect):

    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)  # 假设 Detect 的 __init__ 接受 nc 和 ch

        self.seg_nc = nc
        self.seg_npr = 64 #32
        inc = ch[1]

        self.DGCS1 = DGCS(inc)  # Input: H/8 x W/8 x C -> H/8 x W/8 x 64
        self.DGCS2 = DGCS(inc)
        self.DGCS3 = DGCS(self.seg_npr // 2)
        self.upsample1 = nn.ConvTranspose2d(self.seg_npr, self.seg_npr // 2, 2, 2, 0, bias=True)  # -> H/4 x W/4 x 32
        self.upsample2 = nn.ConvTranspose2d(self.seg_npr, self.seg_npr // 2, 2, 2, 0, bias=True)  # -> H/2 x W/2 x 16
        self.cv = Conv(self.seg_npr // 2, self.seg_npr // 4, k=3)  # -> H/2 x W/2 x 16
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # -> H x W x 16
        self.cv_pred = Conv(self.seg_npr // 4, self.seg_nc + 1, k=1)  # -> H x W x (seg_nc+1)

        # Sigmoid 通常在计算损失时与 BCEWithLogitsLoss 结合使用，这里可以不加
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature_map = x[1]  # 通常来自 layer 28
        f1 = self.DGCS1(feature_map)
        f2 = self.DGCS2(torch.cat([x[0], self.upsample1(f1)], 1))
        f3 = self.DGCS3(self.upsample2(f2))
        f4 = self.cv(self.upsample3(f3))
        p = self.cv_pred(f4)

        return p

class Detect_v2(Detect):

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__(nc=nc, ch=ch)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))  # channels
        for i in range(len(self.cv2)):
            self.cv2[i][1] = DGCS(c2)

########################################################################################################################

#################################################### PConv #############################################################
class PSConv(nn.Module):
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''

    def __init__(self, c1, c2, k, s):
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))
########################################################################################################################

########################################################################################################################
class PSConv_Same(nn.Module):
    def __init__(self, c1, c2, k, s=1): # s should likely always be 1 here
        super().__init__()
        # p = [(k // 2, k // 2, 0, 0), # Pad left/right for horizontal kernel
        #      (0, 0, k // 2, k // 2)] # Pad top/bottom for vertical kernel
             # Using symmetric padding here simplifies maintaining size with k//2
             # Original PSConv used asymmetric padding + ZeroPad layer
             # This version uses Conv's padding argument for simplicity

        # c_ = c2 // 4
        # self.cv_h1 = Conv(c1, c_, (1, k), s=s, p=(0, k // 2)) # Horizontal Conv 1 (e.g., center-right) - adjust padding if needed
        # self.cv_h2 = Conv(c1, c_, (1, k), s=s, p=(0, k // 2)) # Horizontal Conv 2 (e.g., center-left) - needs input manipulation or different padding/kernel?
        # self.cv_v1 = Conv(c1, c_, (k, 1), s=s, p=(k // 2, 0)) # Vertical Conv 1 (e.g., center-down)
        # self.cv_v2 = Conv(c1, c_, (k, 1), s=s, p=(k // 2, 0)) # Vertical Conv 2 (e.g., center-up)

        # Note: Replicating exact PSConv directional effect with standard padding needs care.
        # Or, stick to original PSConv structure but change final 'cat' layer:
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 1, s=1, p=0) # Changed final kernel to 1x1

    def forward(self, x):
        # Simplified conceptual forward using 1x1 fusion conv:
        # Need to implement the 4-branch logic with padding as in original PSConv
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        fused = torch.cat([yw0, yw1, yh0, yh1], dim=1)

        return self.cat(fused)[:, :, :-1, :-1] # Output has same H, W as input 'x'


class SpaceToDepth(nn.Module):
    def __init__(self, block_size=2):
        super().__init__()
        assert block_size == 2, "Only block_size=2 implemented for simplicity"
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # Reshape
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # Permute axes
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # Reshape to final
        return x

class PS2SPD(nn.Module):
    def __init__(self, c1, c2, ps_k=3, final_conv_k=3, c_inter_min=16):
        super().__init__()
        c_intermediate = max(c_inter_min, c1)
        self.ps_enhance = PSConv_Same(c1, c_intermediate, k=ps_k, s=1) # Needs proper implementation
        self.spd = SPDConv(c_intermediate, c_intermediate * 4)
        self.final_conv = Conv(c_intermediate * 4, c2, k=final_conv_k, s=1) # Stride 1 conv

    def forward(self, x):

        x = self.ps_enhance(x)
        x = self.spd(x)
        x = self.final_conv(x)
        return x
########################################################################################################################

######################################## SEAM start ########################################

class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class SEAM(nn.Module):
    def __init__(self, c1, c2, n, reduction=16):
        super(SEAM, self).__init__()
        if c1 != c2:
            c2 = c1
        self.DCovN = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1, groups=c2),
                    nn.GELU(),
                    nn.BatchNorm2d(c2)
                )),
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
                nn.GELU(),
                nn.BatchNorm2d(c2)
            ) for i in range(n)]
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

        self._initialize_weights()
        # self.initialize_layer(self.avg_pool)
        self.initialize_layer(self.fc)


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.DCovN(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def initialize_layer(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

def DcovN(c1, c2, depth, kernel_size=3, patch_size=3):
    dcovn = nn.Sequential(
        nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size),
        nn.SiLU(),
        nn.BatchNorm2d(c2),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=kernel_size, stride=1, padding=1, groups=c2),
                nn.SiLU(),
                nn.BatchNorm2d(c2)
            )),
            nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
            nn.SiLU(),
            nn.BatchNorm2d(c2)
        ) for i in range(depth)]
    )
    return dcovn

class MultiSEAM(nn.Module):
    def __init__(self, c1, c2, depth, kernel_size=3, patch_size=[1, 3, 5], reduction=16):
        super(MultiSEAM, self).__init__()
        if c1 != c2:
            c2 = c1
        self.DCovN0 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[0])
        self.DCovN1 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[1])
        self.DCovN2 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[2])
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        print("111")
        y0 = self.DCovN0(x)
        print("222")
        y1 = self.DCovN1(x)
        y2 = self.DCovN2(x)
        y0 = self.avg_pool(y0).view(b, c)
        y1 = self.avg_pool(y1).view(b, c)
        y2 = self.avg_pool(y2).view(b, c)
        y4 = self.avg_pool(x).view(b, c)
        y = (y0 + y1 + y2 + y4) / 4
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)

######################################## SEAM end ########################################

class Detect_SEAM(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), SEAM(c2, c2, 1, 16), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), SEAM(c3, c3, 1, 16), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

class Detect_MultiSEAM(Detect_SEAM):
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), MultiSEAM(c2, c2, 1), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), MultiSEAM(c3, c3, 1), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
