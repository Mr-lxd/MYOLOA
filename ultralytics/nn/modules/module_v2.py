import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


######################################## Cross-Scale Mutil-Head Self-Attention start ########################################

class CSMHSA(nn.Module):
    def __init__(self, n_dims, heads=8):
        super(CSMHSA, self).__init__()

        self.heads = heads
        self.query = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(n_dims[0], n_dims[1], kernel_size=1)
        )
        self.key = nn.Conv2d(n_dims[1], n_dims[1], kernel_size=1)
        self.value = nn.Conv2d(n_dims[1], n_dims[1], kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_high, x_low = x
        n_batch, C, width, height = x_low.size()
        q = self.query(x_high).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x_low).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x_low).view(n_batch, self.heads, C // self.heads, -1)
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)
        attention = self.softmax(content_content)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)
        return out

######################################## Cross-Scale Mutil-Head Self-Attention end ########################################

######################################## Re-CalibrationFPN end ########################################

def Upsample(x, size, align_corners=False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)


class SBA(nn.Module):

    def __init__(self, inc, input_dim=64):
        super().__init__()

        self.input_dim = input_dim

        self.d_in1 = Conv(input_dim // 2, input_dim // 2, 1)
        self.d_in2 = Conv(input_dim // 2, input_dim // 2, 1)

        self.conv = Conv(input_dim, input_dim, 3)
        self.fc1 = nn.Conv2d(inc[1], input_dim // 2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(inc[0], input_dim // 2, kernel_size=1, bias=False)

        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        H_feature, L_feature = x

        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)

        g_L_feature = self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)

        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)

        L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * Upsample(g_H_feature * H_feature,
                                                                                       size=L_feature.size()[2:],
                                                                                       align_corners=False)
        H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * Upsample(g_L_feature * L_feature,
                                                                                       size=H_feature.size()[2:],
                                                                                       align_corners=False)

        H_feature = Upsample(H_feature, size=L_feature.size()[2:])
        out = self.conv(torch.cat([H_feature, L_feature], dim=1))
        return out


######################################## Re-CalibrationFPN end ########################################


##################################################### LXD #############################################################
class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SelfAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.attn = nn.Parameter(torch.randn(1, out_channels, 1, 1))

    def forward(self, x):
        x = self.conv(x)
        return x * torch.sigmoid(self.attn)

class DynamicResidualBlock(nn.Module):
    def __init__(self, dim):
        super(DynamicResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.attn1 = SelfAttention(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.attn2 = SelfAttention(dim, dim, 1)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.attn1(out)
        out = self.conv2(out)
        out = self.attn2(out)
        out = out * self.scale + residual
        return out
##################################################### LXD #############################################################

######################################## Semantics and Detail Infusion start ########################################
class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)

class SDI(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # self.convs = nn.ModuleList([nn.Conv2d(channel, channels[0], kernel_size=3, stride=1, padding=1) for channel in channels])
        self.convs = nn.ModuleList([GSConv(channel, channels[0]) for channel in channels])

    def forward(self, xs):
        ans = torch.ones_like(xs[0])
        target_size = xs[0].shape[2:]
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[-1]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[-1]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                      mode='bilinear', align_corners=True)
            ans = ans * self.convs[i](x)
        return ans

######################################## Semantics and Detail Infusion end ########################################

######################################## Dynamic Group Convolution Shuffle Transformer start ########################################
class DGCST(nn.Module):
    # Dynamic Group Convolution Shuffle Transformer
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

class DGCST_v2(nn.Module):
    # Dynamic Group Convolution Shuffle Transformer
    def __init__(self, c1, c2) -> None:
        super().__init__()

        self.c = c2 // 4
        self.gconv = Conv(self.c, self.c, g=self.c)
        self.pwconv1 = Conv(self.c, self.c, k=1)
        self.conv1 = Conv(c1, c2, 1)
        self.conv2 = nn.Sequential(
            Conv(c2, c2, 1),
            Conv(c2, c2, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = torch.split(x, [self.c, x.size(1) - self.c], 1)

        x1 = self.gconv(x1)
        x1 = self.pwconv1(x1)

        # shuffle
        b, n, h, w = x1.size()
        b_n = b * n // 2
        y = x1.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        y = torch.cat((y[0], y[1]), 1)

        x = torch.cat([y, x2], 1)
        return x + self.conv2(x)

class DGCST_v3(nn.Module):
    # Dynamic Group Convolution Shuffle Transformer
    def __init__(self, c1, c2) -> None:
        super().__init__()

        self.c = c2 // 4
        self.gconv = Conv(self.c, self.c, g=self.c)
        self.conv1 = Conv(c1, c2, 1)
        self.conv2 = nn.Sequential(
            Conv(c2, c2 // 2, 1),
            DynamicInceptionDWConv2d(c2 // 2, 1, 5),
            Conv(c2 // 2, c2, 1)
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

class DGCST_v4(nn.Module):
    # Dynamic Group Convolution Shuffle Transformer
    def __init__(self, c1, c2) -> None:
        super().__init__()

        self.c = c2 // 4
        self.gconv = Conv(self.c, self.c, g=self.c)
        self.conv1 = Conv(c1, c2, 1)
        self.conv2 = nn.Sequential(
            Conv(c2, c2 // 2, 1),
            DynamicInceptionDWConv2d(c2 // 2, 1, 5),
            Conv(c2 // 2, c2, 1)
        )

    def forward(self, x):
        x_1 = self.conv1(x)

        x1 = self.gconv(x_1)

        # shuffle
        b, n, h, w = x1.size()
        b_n = b * n // 2
        y = x1.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        y = torch.cat((y[0], y[1]), 1)

        x = torch.cat([y, x], 1)
        return x + self.conv2(x)

######################################## Dynamic Group Convolution Shuffle Transformer end ########################################

######################################## LXD ########################################
class ShuffledGroupConv(nn.Module):
    """
    Applies a Grouped Convolution followed by a Channel Shuffle operation.
    Assumes the number of groups equals the number of channels (Depthwise-like).
    """
    def __init__(self, channels, k=3):
        super().__init__()
        # 分组卷积，组数等于通道数 (类似于深度卷积)
        # padding=k//2 保证了当 k 为奇数时，空间尺寸不变
        self.gconv = Conv(channels, channels, k=k, p=k//2, g=channels)

    def forward(self, x):
        x = self.gconv(x)

        b, n, h, w = x.size() # 获取维度: 批量, 通道, 高, 宽

        b_n = b * n // 2 # 计算 reshape 需要的维度

        # 将通道分为两组并展平空间维度
        y = x.reshape(b_n, 2, h * w)
        # 交换维度，将“组内索引”(大小为2)的维度提前
        y = y.permute(1, 0, 2)
        # 恢复空间维度，分离批量维度
        # 使用 -1 让 PyTorch 自动推断批量大小 b，更健壮
        y = y.reshape(2, -1, n // 2, h, w)
        # 沿着通道维度(dim=1)拼接两部分，完成混洗
        y = torch.cat((y[0], y[1]), dim=1)

        return y
######################################## LXD ########################################

######################################## LXD ########################################
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
class Concat_SPD(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, inc, out, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension
        self.downsample = SPDConv(inc, out)

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        # for i, tensor in enumerate(x):
        #     print(f" {i} : {tensor.shape} ")
        # print(f"{self.d}")
        x[1] = self.downsample(x[1])
        return torch.cat(x, self.d)
######################################## LXD ########################################