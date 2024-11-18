import torch
import torch.nn as nn
import torch.nn.functional as F
from mods.bricks import BasicConv2d, CBAM, dwt2d
import math

class DGT(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, wavelet='haar', n_fft=256, hop_length=None):
        super(DGT, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_in = BasicConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_down1 = nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)
        self.conv_down2 = nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)
        self.conv_wt = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.conv_edge = nn.Conv2d(1, out_channels, kernel_size=3, padding=1)
        self.conv_up = BasicConv2d(2, out_channels, kernel_size=3, padding=1)
        self.conv_combine = nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_out = BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.wavelet = wavelet
        self.cbam = CBAM(out_channels, reduction, kernel_size=7)
        
        # Multi-Scale Gabor
        self.theta_values = [0, math.pi/4, math.pi/2, 3*math.pi/4]
        self.sigma = 4.0
        self.lambd = 10.0
        self.gamma = 0.5
        self.psi = 0

    def Gabor(self, x, kernel_size=7):
        gabor_results = []
        for theta in self.theta_values:
            gabor_filter = self.create_gabor_kernel(kernel_size, self.sigma, theta, self.lambd, self.gamma, self.psi, x.device)
            gabor_result = F.conv2d(x, gabor_filter, padding=kernel_size // 2)
            gabor_results.append(gabor_result)
        
        # Averaging to achieve orientation invariance
        gabor_avg = torch.mean(torch.stack(gabor_results, dim=0), dim=0)
        # Normalized Gabor output
        gabor = torch.sigmoid(gabor_avg)
        return gabor

    def create_gabor_kernel(self, kernel_size, sigma, theta, lambd, gamma, psi, device):
        theta = torch.tensor(theta, device=device)
        psi = torch.tensor(psi, device=device)
        x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device)
        y = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device)
        y, x = torch.meshgrid(y, x)
        x_theta = x * torch.cos(theta) + y * torch.sin(theta)
        y_theta = -x * torch.sin(theta) + y * torch.cos(theta)
        gb = torch.exp(-0.5 * (x_theta**2 / sigma**2 + y_theta**2 / (sigma / gamma)**2)) * \
             torch.cos(2 * math.pi * x_theta / lambd + psi)
        return gb.unsqueeze(0).unsqueeze(0)

    def wavelet_transform(self, x):
        B, C, H, W = x.shape
        LL, LH, HL, HH = [], [], [], []
        for i in range(B):
            for j in range(C):
                ll, lh, hl, hh = dwt2d(x[i, j].unsqueeze(0).unsqueeze(0), self.wavelet)
                LL.append(ll.squeeze())
                LH.append(lh.squeeze())
                HL.append(hl.squeeze())
                HH.append(hh.squeeze())
        LL = torch.stack(LL).view(B, C, H // 2, W // 2)
        LH = torch.stack(LH).view(B, C, H // 2, W // 2)
        HL = torch.stack(HL).view(B, C, H // 2, W // 2)
        HH = torch.stack(HH).view(B, C, H // 2, W // 2)
        return LL, LH, HL, HH

    def forward(self, x):
        x = self.conv_in(x)
        X = self.upsample(x)
        x_in = self.conv_down1(x)
        X_in = self.conv_down2(X)
        cA, cH, cV, cD = self.wavelet_transform(X_in)
        edge_features = self.conv_edge(torch.abs(cH) + torch.abs(cV) + torch.abs(cD))
        
        # Using improved multi-scale direction-invariant Gabor filtering
        Gabor = self.Gabor(x_in)
        
        wt = self.conv_wt(cA)
        fusion = torch.cat([wt, Gabor], dim=1)
        fusion = self.conv_up(fusion)
        fusion = self.conv_combine(torch.cat([fusion, edge_features * (x + 1)], dim=1))
        
        # Use CBAM to enhance attention on fusion features
        output = self.cbam(fusion)
        out = self.conv_out(output)
        return out


#  Multi-Scale Domain Transform
class MSDT(nn.Module):
    def __init__(self, in_channels, wavelet = 'haar'):
        super(MSDT, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = BasicConv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = BasicConv2d(in_channels, in_channels, 3, padding=1)
        self.conv3 = BasicConv2d(in_channels, in_channels, 3, padding=1)
        self.conv7 = BasicConv2d(in_channels, in_channels, 3, padding=1)
        self.conv4 = BasicConv2d(in_channels, in_channels, 3, padding=1)
        self.conv5 = BasicConv2d(2 * in_channels, 2 * in_channels, 3, padding=1)
        self.conv6 = BasicConv2d(3 * in_channels, 3 * in_channels, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * in_channels, 2 * in_channels, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * in_channels, 3 * in_channels, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4 * in_channels, 4 * in_channels, 3, padding=1)
        self.conv_4 = BasicConv2d(4 * in_channels, 4 * in_channels, 3, padding=1)
        self.conv_5 = nn.Conv2d(4 * in_channels, in_channels, 1)
        self.conv_6 = nn.Conv2d(in_channels, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.wavelet = wavelet
        self.conv_in = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.fusion_conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

    def wavelet_transform(self, x):
        # wt
        B, C, H, W = x.shape
        LL, LH, HL, HH = [], [], [], []

        for i in range(B):
            for j in range(C):
                ll, lh, hl, hh = dwt2d(x[i, j].unsqueeze(0).unsqueeze(0), self.wavelet)
                LL.append(ll.squeeze())
                LH.append(lh.squeeze())
                HL.append(hl.squeeze())
                HH.append(hh.squeeze())

        LL = torch.stack(LL).view(B, C, H // 2, W // 2)
        LH = torch.stack(LH).view(B, C, H // 2, W // 2)
        HL = torch.stack(HL).view(B, C, H // 2, W // 2)
        HH = torch.stack(HH).view(B, C, H // 2, W // 2)
        return LL, LH, HL, HH

    def forward(self, x1, x2, x3, x4):

        x4_1 = x4
        x3_1 = self.conv1(self.upsample(x4)) + x3
        x2_1 = self.conv2(self.upsample(x3_1)) * self.conv3(self.upsample(x3)) + x2
        x1_1 = self.conv3(self.upsample(x2_1)) * self.conv7(self.upsample(x2)) + x1

        # Further fusion of features
        x3_2 = torch.cat((x3_1, self.conv4(self.upsample(x4_1))), 1)
        x3_2 = self.conv_concat2(x3_2)

        x2_2 = torch.cat((x2_1, self.conv5(self.upsample(x3_2))), 1)
        x2_2 = self.conv_concat3(x2_2)

        x1_2 = torch.cat((x1_1, self.conv6(self.upsample(x2_2))), 1)
        x1_2 = self.conv_concat4(x1_2)

        out = self.conv_4(x1_2)
        fused_features = self.conv_5(out)
        fused_feature = self.conv_6(fused_features)

        X = self.upsample(fused_feature)
        X = self.conv_in(X)

        # Specialized treatment
        cA, cH, cV, cD = self.wavelet_transform(X)
        edge_feature = torch.abs(cH) + torch.abs(cV) + torch.abs(cD)
        out = self.fusion_conv(torch.cat([cA, fused_feature - edge_feature], dim=1))

        return fused_features, out