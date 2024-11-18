import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def dwt2d(x, wavelet):
    # Haar wavelet filters
    if wavelet == 'haar':
        h0 = torch.tensor([1/2**0.5, 1/2**0.5], device=x.device).view(1, 1, 1, -1)
        h1 = torch.tensor([1/2**0.5, -1/2**0.5], device=x.device).view(1, 1, 1, -1)
        g0 = torch.tensor([1/2**0.5, 1/2**0.5], device=x.device).view(1, 1, -1, 1)
        g1 = torch.tensor([1/2**0.5, -1/2**0.5], device=x.device).view(1, 1, -1, 1)
    else:
        raise NotImplementedError("Only Haar wavelet is implemented.")

    # Convolve and downsample
    LL = F.conv2d(F.conv2d(x, h0, stride=1), g0, stride=1)[:, :, ::2, ::2]
    LH = F.conv2d(F.conv2d(x, h0, stride=1), g1, stride=1)[:, :, ::2, ::2]
    HL = F.conv2d(F.conv2d(x, h1, stride=1), g0, stride=1)[:, :, ::2, ::2]
    HH = F.conv2d(F.conv2d(x, h1, stride=1), g1, stride=1)[:, :, ::2, ::2]
    
    return LL, LH, HL, HH


class FourGrad(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        diag1 = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        diag2 = torch.tensor([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('diag1', diag1)
        self.register_buffer('diag2', diag2)

        # Laplacian kernel for regularization
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('laplacian_kernel', laplacian_kernel)

    def forward(self, x):
        b, c, m, n = x.shape

        sobel_x = self.sobel_x.expand(c, 1, 3, 3).to(x.device)
        sobel_y = self.sobel_y.expand(c, 1, 3, 3).to(x.device)
        diag1 = self.diag1.expand(c, 1, 3, 3).to(x.device)
        diag2 = self.diag2.expand(c, 1, 3, 3).to(x.device)
        laplacian_kernel = self.laplacian_kernel.expand(c, 1, 3, 3).to(x.device)

        x1 = F.conv2d(x, sobel_x, padding=1, groups=c)
        x2 = F.conv2d(x, sobel_y, padding=1, groups=c)
        x3 = F.conv2d(x, diag1, padding=1, groups=c)
        x4 = F.conv2d(x, diag2, padding=1, groups=c)

        # Apply Laplacian regularization to each gradient map
        x1_reg = F.conv2d(x1, laplacian_kernel, padding=1, groups=c)
        x2_reg = F.conv2d(x2, laplacian_kernel, padding=1, groups=c)
        x3_reg = F.conv2d(x3, laplacian_kernel, padding=1, groups=c)
        x4_reg = F.conv2d(x4, laplacian_kernel, padding=1, groups=c)

        variance = torch.var(torch.stack([x1_reg, x2_reg, x3_reg, x4_reg], dim=0), dim=0)

        return x1_reg, x2_reg, x3_reg, x4_reg, variance
    

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Average-pool attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        # Max-pool attention
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        # Sum up and apply sigmoid
        out = avg_out + max_out
        return torch.sigmoid(out) * x
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate and apply convolution
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return torch.sigmoid(out) * x


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)
        self.conv = BasicConv2d(2 * in_planes, in_planes, 3, 1, 1)

    def forward(self, x):
        # Apply channel attention
        x = self.channel_attention(x)
        # Apply spatial attention
        x = self.spatial_attention(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, channel):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        weight = self.sigmoid(self.conv(x))
        return weight * x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = F.adaptive_avg_pool2d(x, 1)
        avg_out = self.fc2(F.relu(self.fc1(avg_out)))
        return x * self.sigmoid(avg_out)