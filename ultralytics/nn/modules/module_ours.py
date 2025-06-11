import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from .conv import Conv
from .head import Detect
from einops import rearrange




class SPDC(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x

class DMKDC(nn.Module):
    """ Dynamic Multi-Kernel Depthwise Convolution """

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


class TriPAC(nn.Module):
    """ Tri-Path Adaptive Convolution """
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
        self.convs.append(DMKDC(channels_e1, kernels[0], kernels[0] * 3 + 2))

        # branch 2
        if self.layer_level == 'deep':
            self.convs.append(DMKDC(channels_e2, kernels[1], kernels[1] * 3 + 2))
        # branch 3
        if channels_e3 > 0:
            hidden_channels_e3 = channels_e3 // 2
        self.cv2 = Conv(channels_e3, hidden_channels_e3, 1)
        self.m = nn.ModuleList(TriPGCB(hidden_channels_e3) for _ in range(1))
        self.cv3 = Conv(hidden_channels_e3, channels_e3, 1)

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
#
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

class TriPGCB(nn.Module):
    """ Tri-Path Gated CNN Block """
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

class DGST(nn.Module):
    # Dynamic Group Shuffle Transformer
    def __init__(self, c1, c2) -> None:
        super().__init__()

        self.c = c2 // 4
        self.gconv = Conv(self.c, self.c, g=self.c)
        self.conv1 = Conv(c1, c2, 1)
        self.conv2 = nn.Sequential(
            Conv(c2, c2, 1),
            Conv(c2, c2, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
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
        return x + self.conv2(x)

class Segment_v3(Detect):

    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)

        self.seg_nc = nc
        self.seg_npr = 32 #64
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

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature_map = x[0]
        f1 = self.cv1(feature_map)
        f2 = self.cv2(self.upsample1(f1))
        f3 = self.cv3(self.upsample2(f2))
        f4 = self.cv4(self.upsample3(f3))
        p = self.cv_pred(f4)

        return p
