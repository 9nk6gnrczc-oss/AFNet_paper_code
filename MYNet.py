import torch
import torch.nn as nn
from models.smt import smt_t
from thop import profile
import torch.nn.functional as F

def conv3x3_bn_gelu(in_planes, out_planes, k=3, s=1, p=1, b=False):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
            nn.BatchNorm2d(out_planes),
            nn.GELU(),
            )

class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels  # 初始化深度
        chunk_dim = dim // n_levels
        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])
        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):

            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)
        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out

class S_Attention(nn.Module):
    def __init__(self, dim):
        super(S_Attention, self).__init__()
        self.sa = nn.Conv2d(3 * dim, 1, 7, padding=3,
                            padding_mode='reflect', bias=True)

    def forward(self, x):
        # 多尺度池化
        x_pool1 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x_pool2 = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        x_pool3 = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        # 拼接多尺度池化结果
        x_con = torch.cat([x_pool1, x_pool2, x_pool3], dim=1)
        # 通过卷积层生成空间注意力权重
        sa_att = self.sa(x_con)
        return sa_att

class C_Attention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(C_Attention, self).__init__()
        self.a_avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_dim = dim // reduction
        self.ca = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, padding=0, bias=True),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.a_avg_pool(x)
        ca_att = self.ca(x_gap)
        return ca_att

class CS_Fusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CS_Fusion, self).__init__()
        self.sa = S_Attention(dim)
        self.ca = C_Attention(dim, reduction)
        self.conv_1 = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.saf = SAFM(dim)

    def forward(self, x, y):
        feature = x + y
        ca_feature = self.ca(feature)
        ca_W = self.sigmoid(ca_feature)
        ca = ca_W * feature
        cs_feature = self.sa(ca)
        cs_W = self.sigmoid(cs_feature)
        f_cs_feature = feature + cs_W * x + (1 - cs_W) * y
        result = self.conv_1(f_cs_feature)
        result = self.saf(result)
        return result

class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim / growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x_1 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        edge = self.conv_0(x - x_1)
        x_con = edge + x
        x = self.up(x_con)
        return x


class MYNet(nn.Module):
    def __init__(self):
        super(MYNet, self).__init__()
        self.smt = smt_t()
        # 添加CGAFusion模块
        self.csf_1 = CS_Fusion(dim=256)
        self.csf_2 = CS_Fusion(dim=128)
        self.csf_3 = CS_Fusion(dim=64)
        self.csf_4 = CS_Fusion(dim=32)

        self.dmlp_1 = DMlp(dim=32)
        self.dmlp_2 = DMlp(dim=32)
        self.dmlp_3 = DMlp(dim=32)
        self.dmlp_4 = DMlp(dim=32)

        self.rfc1 = conv3x3_bn_gelu(512, 256, k=1, s=1, p=0)
        self.rfc2 = conv3x3_bn_gelu(256, 128,  k=1, s=1, p=0)
        self.rfc3 = conv3x3_bn_gelu(128, 64,  k=1, s=1, p=0)
        self.rfc4 = conv3x3_bn_gelu(64, 32,  k=1, s=1, p=0)

        self.rfc_1 = conv3x3_bn_gelu(256, 256)
        self.rfc_2 = conv3x3_bn_gelu(128, 128)
        self.rfc_3 = conv3x3_bn_gelu(64, 64)
        self.rfc_4 = conv3x3_bn_gelu(32, 32)

        self.xf_11 = nn.Conv2d(256, 32, kernel_size=1)
        self.xf_22 = nn.Conv2d(128, 32, kernel_size=1)
        self.xf_33 = nn.Conv2d(64, 32, kernel_size=1)

        self.dec_2 = nn.Conv2d(64, 32, kernel_size=1)
        self.dec_3 = nn.Conv2d(64, 32, kernel_size=1)
        self.dec_4 = nn.Conv2d(64, 32, kernel_size=1)


        self.pre_trans1 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.pre_trans2 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.pre_trans3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.pre_trans4 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        rgb_list = self.smt(x)

        r1 = rgb_list[3]  # 512,12
        r2 = rgb_list[2]  # 256,24
        r3 = rgb_list[1]  # 128,48
        r4 = rgb_list[0]  # 64,96

        r1_up = F.interpolate(self.rfc1(r1), size=24, mode='bilinear')  # 256,24
        xf_1 = self.csf_1(r1_up, r2)
        xf_1 = self.rfc_1(xf_1)

        r2_up = F.interpolate(self.rfc2(xf_1), size=48, mode='bilinear')  # 128,48
        xf_2 = self.csf_2(r2_up, r3)
        xf_2 = self.rfc_2(xf_2)

        r3_up = F.interpolate(self.rfc3(xf_2), size=96, mode='bilinear')  # 64,96
        xf_3 = self.csf_3(r3_up, r4)
        xf_3 = self.rfc_3(xf_3)

        r4_up = F.interpolate(self.rfc4(xf_3), size=192, mode='bilinear')  # 32,192
        r4_s = F.interpolate(self.rfc4(r4), size=192, mode='bilinear')  # 32,192
        xf_4 = self.csf_4(r4_up, r4_s)
        xf_4 = self.rfc_4(xf_4)

        xf_11 = self.xf_11(xf_1)
        xf_22 = self.xf_22(xf_2)
        xf_33 = self.xf_33(xf_3)

        xf_1 = self.dmlp_1(xf_11)
        xc_1_2 = torch.cat((xf_1, xf_22), 1)
        df_12 = self.dec_2(xc_1_2)

        xf_2 = self.dmlp_2(df_12)
        xc_2_3 = torch.cat((xf_2, xf_33), 1)
        df_23 = self.dec_3(xc_2_3)

        xf_3 = self.dmlp_3(df_23)
        xc_3_4 = torch.cat((xf_3, xf_4), 1)
        df_34 = self.dec_4(xc_3_4)

        xf_4 = self.dmlp_4(df_34)

        y1 = self.pre_trans1(xf_4)
        y2 = F.interpolate(self.pre_trans2(xf_3), size=384, mode='bilinear')
        y3 = F.interpolate(self.pre_trans3(xf_2), size=384, mode='bilinear')
        y4 = F.interpolate(self.pre_trans4(xf_1), size=384, mode='bilinear')
        return y1, y2, y3, y4

    def load_pre(self, pre_model):
        self.smt.load_state_dict(torch.load(pre_model)['model'])
        print(f"loading pre_model ${pre_model}")


if __name__ == '__main__':
    x = torch.randn(1, 3, 384, 384)
    model = MYNet()  # 确保只传递一个参数
    flops, params = profile(model, (x,))
    print('flops: %.3f G, params: %.3f M' % (flops / 1000000000.0, params / 1000000.0))