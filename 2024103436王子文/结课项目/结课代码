import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Parameter, Softmax
from lib.swin_transformer import SwinTransformer
from models.OctaveConv2 import FirstOctaveCBR, LastOCtaveCBR, OctaveCBR
from models.DCTlayer import MultiSpectralAttentionLayer
from models.exchange_modules import *
import math
from models.GCN import *

class MultiScaleContextEnhancement(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.channels = in_channels
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1),
            nn.ReLU(inplace=True)
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 3, padding=3, dilation=3),
            nn.ReLU(inplace=True))
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 3, padding=5, dilation=5),
            nn.ReLU(inplace=True))
        
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 3, padding=7, dilation=7),
            nn.ReLU(inplace=True))
        
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction, in_channels, 1),
            nn.Sigmoid())
        
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        multi_scale = torch.cat([b1, b2, b3, b4], dim=1)
        
        att = self.att(multi_scale)
        att_multi_scale = multi_scale * att
        
        out = torch.cat([x, att_multi_scale], dim=1)
        out = self.fusion(out)
        
        return out + x

class AdaptiveFrequencyDecomposition(nn.Module):
    def __init__(self, in_channels, ratios=(0.3, 0.5, 0.7)):
        super().__init__()
        self.ratios = ratios
        self.conv_low = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_mid = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_high = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.fusion = nn.Conv2d(in_channels*3, in_channels, 1)
        
    def forward(self, x):
        fft = torch.fft.rfft2(x, norm='ortho')
        magnitude = torch.abs(fft)
        phase = torch.angle(fft)
        
        _, _, h, w = magnitude.shape
        mask_low = torch.zeros((h, w//2+1)).to(x.device)
        mask_mid = torch.zeros((h, w//2+1)).to(x.device)
        mask_high = torch.zeros((h, w//2+1)).to(x.device)
        
        for i in range(h):
            for j in range(w//2+1):
                freq = math.sqrt(i**2 + j**2)
                max_freq = math.sqrt((h/2)**2 + (w/2)**2)
                ratio = freq / max_freq
                
                if ratio < self.ratios[0]:
                    mask_low[i, j] = 1
                elif ratio < self.ratios[1]:
                    mask_mid[i, j] = 1
                else:
                    mask_high[i, j] = 1
        
        low_freq = magnitude * mask_low
        mid_freq = magnitude * mask_mid
        high_freq = magnitude * mask_high
        
        low_complex = low_freq * torch.exp(1j * phase)
        mid_complex = mid_freq * torch.exp(1j * phase)
        high_complex = high_freq * torch.exp(1j * phase)
        
        low_spatial = torch.fft.irfft2(low_complex, s=(h, w), norm='ortho')
        mid_spatial = torch.fft.irfft2(mid_complex, s=(h, w), norm='ortho')
        high_spatial = torch.fft.irfft2(high_complex, s=(h, w), norm='ortho')
        
        low_feat = self.conv_low(low_spatial)
        mid_feat = self.conv_mid(mid_spatial)
        high_feat = self.conv_high(high_spatial)
        
        fused = torch.cat([low_feat, mid_feat, high_feat], dim=1)
        return self.fusion(fused)

class CrossModalContrastiveLearning(nn.Module):
    def __init__(self, embed_dim=128, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim))
        
    def forward(self, rgb_feat, t_feat, d_feat):
        rgb_proj = self.projector(rgb_feat.mean(dim=[2,3]))
        t_proj = self.projector(t_feat.mean(dim=[2,3]))
        d_proj = self.projector(d_feat.mean(dim=[2,3]))
        
        features = torch.stack([rgb_proj, t_proj, d_proj], dim=1)
        similarity = torch.matmul(features, features.transpose(1,2)) / self.temperature
        
        batch_size = similarity.size(0)
        labels = torch.arange(0, 3 * batch_size, 3, device=similarity.device)
        
        loss_fct = nn.CrossEntropyLoss()
        contrastive_loss = loss_fct(similarity.view(-1, 3), labels)
        
        return contrastive_loss

class BoundaryAwareRefinement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, 1, 1))
        
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels+1, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1))
        
    def forward(self, x):
        edge = self.edge_conv(x)
        edge_sigmoid = torch.sigmoid(edge)
        
        refined = torch.cat([x, edge_sigmoid], dim=1)
        refined = self.refine_conv(refined)
        
        return refined + x, edge_sigmoid

class DynamicFeatureExchange(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//4, 1),
            nn.ReLU(),
            nn.Conv2d(channel//4, 3, 1),
            nn.Softmax(dim=1))
        
    def forward(self, rgb, t, d):
        combined = rgb + t + d
        weights = self.gate(combined)
        
        w_rgb, w_t, w_d = weights[:,0:1], weights[:,1:2], weights[:,2:3]
        
        new_rgb = w_rgb * rgb + (1 - w_rgb) * (t + d)/2
        new_t = w_t * t + (1 - w_t) * (rgb + d)/2
        new_d = w_d * d + (1 - w_d) * (rgb + t)/2
        
        return new_rgb, new_t, new_d

class FeatureCalibration(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.calibrate = nn.Sequential(
            nn.Conv2d(channel*3, channel, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.Sigmoid())
        
    def forward(self, rgb, t, d):
        combined = torch.cat([rgb, t, d], dim=1)
        weight = self.calibrate(combined)
        calibrated_rgb = rgb * weight + rgb
        calibrated_t = t * weight + t
        calibrated_d = d * weight + d
        return calibrated_rgb, calibrated_t, calibrated_d

class MultiLevelSupervision(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels[0], 1, 1)
        self.conv2 = nn.Conv2d(channels[1], 1, 1)
        self.conv3 = nn.Conv2d(channels[2], 1, 1)
        self.conv4 = nn.Conv2d(channels[3], 1, 1)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        
    def forward(self, x1, x2, x3, x4):
        s1 = self.conv1(x1)
        s2 = self.upsample2(self.conv2(x2))
        s3 = self.upsample4(self.conv3(x3))
        s4 = self.upsample8(self.conv4(x4))
        return s1, s2, s3, s4

class Gate(nn.Module):
    def __init__(self, in_plane):
        super(Gate, self).__init__()
        self.gate1 = nn.Conv3d(in_plane,in_plane, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.gate2 = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, fea):
        gate = torch.sigmoid(self.gate1(fea))
        gate_fea = fea * gate + fea
        gate_fea = gate_fea.permute(0, 2, 1, 3, 4)
        gate_fea = torch.squeeze(self.gate2(gate_fea), dim=1)

        return gate_fea


class Mnet(nn.Module):
    def __init__(self, pretrained=False):
        super(Mnet, self).__init__()
        self.swin1 = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.ReLU = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.gate4 = Gate(512)
        self.gate3 = Gate(256)
        self.gate2 = Gate(128)
        self.gate1 = Gate(64)

        self.CBAM1 = CBAMLayer(64)
        self.CBAM2 = CBAMLayer(128)
        self.CBAM3 = CBAMLayer(256)
        self.CBAM4 = CBAMLayer(512)

        self.mscem1 = MultiScaleContextEnhancement(64)
        self.mscem2 = MultiScaleContextEnhancement(128)
        self.mscem3 = MultiScaleContextEnhancement(256)
        self.mscem4 = MultiScaleContextEnhancement(512)
        
        self.afd1 = AdaptiveFrequencyDecomposition(64)
        self.afd2 = AdaptiveFrequencyDecomposition(128)
        self.afd3 = AdaptiveFrequencyDecomposition(256)
        self.afd4 = AdaptiveFrequencyDecomposition(512)
        
        self.bar1 = BoundaryAwareRefinement(64)
        self.bar2 = BoundaryAwareRefinement(128)
        self.bar3 = BoundaryAwareRefinement(256)
        self.bar4 = BoundaryAwareRefinement(512)
        
        self.dfe1 = DynamicFeatureExchange(64)
        self.dfe2 = DynamicFeatureExchange(128)
        self.dfe3 = DynamicFeatureExchange(256)
        self.dfe4 = DynamicFeatureExchange(512)
        
        self.fc1 = FeatureCalibration(64)
        self.fc2 = FeatureCalibration(128)
        self.fc3 = FeatureCalibration(256)
        self.fc4 = FeatureCalibration(512)
        
        self.mls = MultiLevelSupervision([64, 128, 256, 512])
        
        self.cmcl = CrossModalContrastiveLearning(128)

        self.de_conv4 = Conv(512, 256, 3, bn=False, relu=False)
        self.de_conv3 = Conv(256, 128, 3, bn=False, relu=False)
        self.de_conv2 = Conv(128, 64, 3, bn=False, relu=False)
        self.de_conv1 = Conv(64, 32, 3, bn=False, relu=False)

        self.interaction4 =Interaction(in_channels=1024, out_channels=512)
        self.interaction3 = Interaction(in_channels=512, out_channels=256)
        self.interaction2 = Interaction(in_channels=256, out_channels=128)
        self.interaction1 =Interaction(in_channels=128, out_channels=64)

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_1_d = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_1_t = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

        self.conv4_vdt = nn.Sequential(
            FirstOctaveCBR(in_channels=512, out_channels=512, kernel_size=(1,1), padding=0, bias=True),
        )
        self.conv3_vdt = nn.Sequential(
            FirstOctaveCBR(in_channels=256, out_channels=256, kernel_size=(1,1), padding=0, bias=True),
        )
        self.conv2_vdt = nn.Sequential(
            FirstOctaveCBR(in_channels=128, out_channels=128, kernel_size=(1,1), padding=0, bias=True),
        )
        self.conv1_vdt = nn.Sequential(
            FirstOctaveCBR(in_channels=64, out_channels=64, kernel_size=(1,1), padding=0, bias=True),
        )


        self.conv4_vdt2 = nn.Sequential(
            OctaveCBR(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=True),
            LastOCtaveCBR(in_channels=512, out_channels=256, kernel_size=(1,1), padding=0, bias=True),

        )
        self.conv3_vdt2 = nn.Sequential(
            OctaveCBR(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, bias=True),
            LastOCtaveCBR(in_channels=256, out_channels=128, kernel_size=(1,1), padding=0, bias=True),

        )
        self.conv2_vdt2 = nn.Sequential(
            OctaveCBR(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=True),
            LastOCtaveCBR(in_channels=128, out_channels=64, kernel_size=(1,1), padding=0, bias=True),

        )
        self.conv1_vdt2 = nn.Sequential(
            OctaveCBR(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=True),
            LastOCtaveCBR(in_channels=64, out_channels=32, kernel_size=(1,1), padding=0, bias=True),
        )

        self.LF_conv4 = Conv(256, 256, 3, bn=True, relu=True)
        self.LF_conv3 = Conv(128, 128, 3, bn=True, relu=True)
        self.LF_conv2 = Conv(64, 64, 3, bn=True, relu=True)
        self.LF_conv1 = Conv(32, 32, 3, bn=True, relu=True)
        self.DCTatt_L4 = MultiSpectralAttentionLayer(256, 6, 6,  reduction=8, freq_sel_method = 'low4')
        self.DCTatt_L3 = MultiSpectralAttentionLayer(128, 12, 12,  reduction=8, freq_sel_method = 'low4')
        self.DCTatt_L2 = MultiSpectralAttentionLayer(64, 24, 24,  reduction=8, freq_sel_method = 'low4')
        self.DCTatt_L1 = MultiSpectralAttentionLayer(32, 48, 48,  reduction=8, freq_sel_method = 'low4')

        self.HF_conv4 = Conv(256, 256, 3, bn=True, relu=True)
        self.HF_conv3 = Conv(128, 128, 3, bn=True, relu=True)
        self.HF_conv2 = Conv(64, 64, 3, bn=True, relu=True)
        self.HF_conv1 = Conv(32, 32, 3, bn=True, relu=True)
        self.DCTatt_H4 = MultiSpectralAttentionLayer(256, 12, 12,  reduction=8, freq_sel_method = 'hig4')
        self.DCTatt_H3 = MultiSpectralAttentionLayer(128, 24, 24,  reduction=8, freq_sel_method = 'hig4')
        self.DCTatt_H2 = MultiSpectralAttentionLayer(64, 48, 48,  reduction=8, freq_sel_method = 'hig4')
        self.DCTatt_H1 = MultiSpectralAttentionLayer(32, 96, 96,  reduction=8, freq_sel_method = 'hig4')

        self.final_4_vdt_h = nn.Sequential(
            Conv(256, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_4_vdt_l = nn.Sequential(
            Conv(256, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_4_vdt = nn.Sequential(
            Conv(256, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_3_vdt_h = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_3_vdt_l = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_3_vdt = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_2_vdt_h = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_2_vdt_l = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_2_vdt = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_1_vdt_h = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, 1, 1, bn=False, relu=False)
        )
        self.final_1_vdt_l = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, 1, 1, bn=False, relu=False)
        )
        self.final_1_vdt = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, 1, 1, bn=False, relu=False)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        self.edge_pred = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

    def forward(self, rgb, t, d):
        score_list_t, score_PE = self.swin1(t)
        score_list_rgb, score_PE = self.swin1(rgb)
        score_list_d, score_PE = self.swin1(d)

        contrastive_loss = self.cmcl(score_list_rgb[0], score_list_t[0], score_list_d[0])

        x1 = [score_list_rgb[0],score_list_t[0],score_list_d[0]]
        x2 = [score_list_rgb[1],score_list_t[1],score_list_d[1]]
        x3 = [score_list_rgb[2],score_list_t[2],score_list_d[2]]
        x4 = [score_list_rgb[3],score_list_t[3],score_list_d[3]]
        
        x1_rgb, x1_t, x1_d = self.dfe1(x1[0], x1[1], x1[2])
        x1 = [x1_rgb, x1_t, x1_d]
        
        x2_rgb, x2_t, x2_d = self.dfe2(x2[0], x2[1], x2[2])
        x2 = [x2_rgb, x2_t, x2_d]
        
        x3_rgb, x3_t, x3_d = self.dfe3(x3[0], x3[1], x3[2])
        x3 = [x3_rgb, x3_t, x3_d]
        
        x4_rgb, x4_t, x4_d = self.dfe4(x4[0], x4[1], x4[2])
        x4 = [x4_rgb, x4_t, x4_d]
        
        x1_calibrated = self.fc1(x1[0], x1[1], x1[2])
        x1 = list(x1_calibrated)
        
        x2_calibrated = self.fc2(x2[0], x2[1], x2[2])
        x2 = list(x2_calibrated)
        
        x3_calibrated = self.fc3(x3[0], x3[1], x3[2])
        x3 = list(x3_calibrated)
        
        x4_calibrated = self.fc4(x4[0], x4[1], x4[2])
        x4 = list(x4_calibrated)

        x4e = self.interaction4(x4)
        x4e = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x4_) for x4_ in x4e]
        
        x4e = [self.mscem4(x) for x in x4e]
        
        x4e = [self.afd4(x) for x in x4e]

        x3e = [x3[i]+x4e[i] for i in range(3)]
        x3e = self.interaction3(x3e)
        x3e = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(xe_) for xe_ in x3e]
        
        x3e = [self.mscem3(x) for x in x3e]
        
        x3e = [self.afd3(x) for x in x3e]

        x2e = [x2[i] + x3e[i] for i in range(3)]
        x2e = self.interaction2(x2e)
        x2e = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(xe_) for xe_ in x2e]
        
        x2e = [self.mscem2(x) for x in x2e]
        
        x2e = [self.afd2(x) for x in x2e]

        x1e = [x1[i] + x2e[i] for i in range(3)]
        x1e = self.interaction1(x1e)
        x1e = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(xe_) for xe_ in x1e]
        
        x1e = [self.mscem1(x) for x in x1e]
        
        x1e = [self.afd1(x) for x in x1e]

        x4e_v = x4e[0]
        x3e_v = x3e[0]
        x2e_v = x2e[0]
        x1e_v = x1e[0]
        x1e_pred = self.final_1(x1e_v)
        x1e_pred = self.up2(x1e_pred)

        x4e_t = x4e[1]
        x3e_t = x3e[1]
        x2e_t = x2e[1]
        x1e_t = x1e[1]
        x1e_pred_t = self.final_1_t(x1e_t)
        x1e_pred_t = self.up2(x1e_pred_t)

        x4e_d = x4e[2]
        x3e_d = x3e[2]
        x2e_d = x2e[2]
        x1e_d = x1e[2]
        x1e_pred_d = self.final_1_d(x1e_d)
        x1e_pred_d = self.up2(x1e_pred_d)

        x1e_vdt = torch.stack([x1e_t, x1e_v, x1e_d], dim=2)
        x1_vdt = self.gate1(x1e_vdt)
        
        x1_vdt, edge1 = self.bar1(x1_vdt)

        x2e_vdt = torch.stack([x2e_t, x2e_v, x2e_d], dim=2)
        x2_vdt = self.gate2(x2e_vdt)
        
        x2_vdt, edge2 = self.bar2(x2_vdt)

        x3e_vdt = torch.stack([x3e_t, x3e_v, x3e_d], dim=2)
        x3_vdt = self.gate3(x3e_vdt)
        
        x3_vdt, edge3 = self.bar3(x3_vdt)

        x4e_vdt = torch.stack([x4e_t, x4e_v, x4e_d], dim=2)
        x4_vdt = self.gate4(x4e_vdt)
        
        x4_vdt, edge4 = self.bar4(x4_vdt)

        x4e_vdt_h,x4e_vdt_l = self.conv4_vdt(x4_vdt)
        x4e_vdt_h = self.DCTatt_H4(x4e_vdt_h)
        x4e_vdt_l = self.DCTatt_L4(x4e_vdt_l)
        x4e_vdt_h = self.HF_conv4(x4e_vdt_h)
        x4e_vdt_l = self.LF_conv4(x4e_vdt_l)
        x4e_pred_vdt_h = self.final_4_vdt_h(x4e_vdt_h)
        x4e_pred_vdt_l = self.final_4_vdt_l(x4e_vdt_l)
        x4e_vdt = self.conv4_vdt2((x4e_vdt_h,x4e_vdt_l))
        x4s_vdt = self.de_conv4(self.CBAM4(x4_vdt))
        x4e_vdt = x4e_vdt + x4s_vdt
        x4e_vdt = self.up2(x4e_vdt)
        x4e_pred_vdt = self.final_4_vdt(x4e_vdt)

        x3i_vdt = x4e_vdt + x3_vdt
        x3e_vdt_h,x3e_vdt_l = self.conv3_vdt(x3i_vdt)
        x3e_vdt_h = self.DCTatt_H3(x3e_vdt_h)
        x3e_vdt_l = self.DCTatt_L3(x3e_vdt_l)
        x3e_vdt_h = self.HF_conv3(x3e_vdt_h)
        x3e_vdt_l = self.LF_conv3(x3e_vdt_l)
        x3e_pred_vdt_h = self.final_3_vdt_h(x3e_vdt_h)
        x3e_pred_vdt_l = self.final_3_vdt_l(x3e_vdt_l)
        x3e_vdt = self.conv3_vdt2((x3e_vdt_h, x3e_vdt_l))
        x3s_vdt = self.de_conv3(self.CBAM3(x3i_vdt))
        x3e_vdt = x3e_vdt + x3s_vdt
        x3e_vdt = self.up2(x3e_vdt)
        x3e_pred_vdt = self.final_3_vdt(x3e_vdt)

        x2i_vdt = x3e_vdt + x2_vdt
        x2e_vdt_h,x2e_vdt_l = self.conv2_vdt(x2i_vdt)
        x2e_vdt_h = self.DCTatt_H2(x2e_vdt_h)
        x2e_vdt_l = self.DCTatt_L2(x2e_vdt_l)
        x2e_vdt_h = self.HF_conv2(x2e_vdt_h)
        x2e_vdt_l = self.LF_conv2(x2e_vdt_l)
        x2e_pred_vdt_h = self.final_2_vdt_h(x2e_vdt_h)
        x2e_pred_vdt_l = self.final_2_vdt_l(x2e_vdt_l)
        x2e_vdt = self.conv2_vdt2((x2e_vdt_h, x2e_vdt_l))
        x2s_vdt = self.de_conv2(self.CBAM2(x2i_vdt))
        x2e_vdt = x2e_vdt + x2s_vdt
        x2e_vdt = self.up2(x2e_vdt)
        x2e_pred_vdt = self.final_2_vdt(x2e_vdt)

        x1i_vdt = x2e_vdt + x1_vdt
        x1e_vdt_h,x1e_vdt_l = self.conv1_vdt(x1i_vdt)
        x1e_vdt_h = self.DCTatt_H1(x1e_vdt_h)
        x1e_vdt_l = self.DCTatt_L1(x1e_vdt_l)
        x1e_vdt_h = self.HF_conv1(x1e_vdt_h)
        x1e_vdt_l = self.LF_conv1(x1e_vdt_l)
        x1e_pred_vdt_h = self.final_1_vdt_h(x1e_vdt_h)
        x1e_pred_vdt_l = self.final_1_vdt_l(x1e_vdt_l)
        x1e_vdt = self.conv1_vdt2((x1e_vdt_h, x1e_vdt_l))
        x1s_vdt = self.de_conv1(self.CBAM1(x1i_vdt))
        x1e_vdt = x1e_vdt + x1s_vdt
        x1e_vdt = self.up2(x1e_vdt)
        x1e_pred_vdt = self.final_1_vdt(x1e_vdt)
        
        edge_pred = self.edge_pred(x1e_vdt)

        x2e_pred_vdt = self.up2(x2e_pred_vdt)
        x3e_pred_vdt = self.up4(x3e_pred_vdt)
        x4e_pred_vdt = self.up8(x4e_pred_vdt)
        x_pred = [x1e_pred_vdt, x2e_pred_vdt, x3e_pred_vdt, x4e_pred_vdt]

        x1e_pred_vdt_l = self.up4(x1e_pred_vdt_l)
        x2e_pred_vdt_l = self.up8(x2e_pred_vdt_l)
        x3e_pred_vdt_l = self.up16(x3e_pred_vdt_l)
        x4e_pred_vdt_l = self.up32(x4e_pred_vdt_l)
        x_pred_l = [x1e_pred_vdt_l, x2e_pred_vdt_l, x3e_pred_vdt_l, x4e_pred_vdt_l]

        x1e_pred_vdt_h = self.up2(x1e_pred_vdt_h)
        x2e_pred_vdt_h = self.up4(x2e_pred_vdt_h)
        x3e_pred_vdt_h = self.up8(x3e_pred_vdt_h)
        x4e_pred_vdt_h = self.up16(x4e_pred_vdt_h)
        x_pred_h = [x1e_pred_vdt_h, x2e_pred_vdt_h, x3e_pred_vdt_h, x4e_pred_vdt_h]
        
        s1, s2, s3, s4 = self.mls(x1e_vdt, x2e_vdt, x3e_vdt, x4e_vdt)
        
        edge_pred = self.up2(edge_pred)
        
        return {
            'pred_v': x1e_pred,
            'pred_t': x1e_pred_t,
            'pred_d': x1e_pred_d,
            'pred_fused': x_pred,
            'pred_low': x_pred_l,
            'pred_high': x_pred_h,
            'pred_edge': edge_pred,
            'contrastive_loss': contrastive_loss,
            'supervision': [s1, s2, s3, s4],
            'edges': [edge1, edge2, edge3, edge4]
        }

    def load_pretrained_model(self):
        self.swin1.load_state_dict(torch.load('/home/swin_base_patch4_window12_384_22k.pth')['model'],strict=False)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
