import torch
import torch.nn as nn
import torch.nn.functional as F
from mods.bricks import BasicConv2d, SEBlock, FourGrad


# Anisotropic Gradient Feature
class AGF(nn.Module):
    def __init__(self, op_channel, tg_channel):
        super().__init__()
        self.conv_in = BasicConv2d(op_channel, op_channel, 3, 1, 1)
        self.conv_fg = BasicConv2d(op_channel, op_channel, 3, 1, 1)
        
        self.conv_tri1 = BasicConv2d(3 * op_channel, op_channel, 3, 1, 1)
        self.conv_tri2 = BasicConv2d(3 * op_channel, op_channel, 3, 1, 1)
        self.conv_tri3 = BasicConv2d(3 * op_channel, op_channel, 3, 1, 1)
        
        # Adaptive edge enhancement module
        self.edge_attention = SEBlock(op_channel)
        
        self.seq = nn.Sequential(
            nn.Conv2d(op_channel, op_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(op_channel, affine=False)
        )
        
        self.conv_down = nn.Conv2d(op_channel, tg_channel, 3, 1, 1)
        self.conv_out = BasicConv2d(tg_channel, tg_channel, 3, 1, 1)
        
        # Residual Connection Layer
        self.residual = nn.Conv2d(op_channel, tg_channel, 1)
        
        # Adaptive weighting parameters
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def adaptive_weighted_edge_detection(self, out1, out2, out3, out4):
        epsilon = 1e-6
        weights = torch.softmax(torch.tensor([0.1, 0.3, 0.3, 0.3]).to(out1.device), dim=0).view(1, 4, 1, 1)
        weights = weights.repeat(1, out1.shape[1] // 4, 1, 1)
        
        edge_map = torch.sqrt((out1**2 + out2**2 + out3**2 + out4**2 + epsilon) * weights)
        return edge_map

    def forward(self, x):
        fg = FourGrad()
        x_in = self.conv_in(x)

        out1, out2, out3, out4, var1 = fg(x_in)
        edge = self.adaptive_weighted_edge_detection(out1, out2, out3, out4)
        FG = self.edge_attention(self.conv_fg(edge))
        
        var = self.seq(var1)
        similarity = F.cosine_similarity(var, FG, dim=1).unsqueeze(1)
        enhanced_feature = (self.alpha * similarity * FG) + (self.beta * (1 - similarity) * var)
        
        # Residual connection enhancement
        enhanced_feature = self.conv_down(enhanced_feature) + self.residual(x_in)
        
        # Multi-channel fusion of output features
        out = self.conv_out(enhanced_feature) + enhanced_feature * self.residual(x_in)
        return out


#  Multi-Scale Anisotropic Gradient
class MSAG(nn.Module):
    def __init__(self, in_channels):
        super(MSAG, self).__init__()
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
        self.conv_in = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.fusion_conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2, x3, x4):

        x4_1 = x4
        x3_1 = self.conv1(self.upsample(x4)) + x3
        x2_1 = self.conv2(self.upsample(x3_1)) * self.conv3(self.upsample(x3)) + x2
        x1_1 = self.conv3(self.upsample(x2_1)) * self.conv7(self.upsample(x2)) + x1

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

        return fused_features, X
