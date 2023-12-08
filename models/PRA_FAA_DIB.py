import torchvision
import torch
from torch import nn
from torch.nn import init
from models.utils import pooling
import torch.nn.functional as F


# upsample encoder
class GEN(nn.Module):
    def __init__(self, in_feat_dim, out_img_dim, config, **kwargs):
        super().__init__()

        self.in_feat_dim = in_feat_dim
        self.out_img_dim = out_img_dim

        self.conv0 = nn.Conv2d(self.in_feat_dim, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
        self.conv4 = nn.Conv2d(32, self.out_img_dim, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)

        self.up = nn.Upsample(scale_factor=2)

        self.bn = nn.BatchNorm2d(64)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.up(x)
        x = self.conv1(x)
        x = self.relu(x)

        x = self.up(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = self.up(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = self.up(x)
        x = self.conv4(x)
        x = torch.tanh(x)

        return x

# FAA (simple vision)
class FAA(nn.Module):

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 sub_sample=True,
                 bn_layer=True):
        super(FAA, self).__init__()
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            # 进行压缩得到channel个数
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c,  h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        # print(f.shape)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

# PRA
class PRA(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(PRA, self).__init__()

        self.h = h
        self.w = w
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)
        self.SFT_shift_conv0 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1,
                                         stride=1, bias=False)
        self.SFT_shift_conv1 = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1,
                                         stride=1, bias=False)
        self.SFT_scale_conv0 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1,
                                         stride=1, bias=False)
        self.SFT_scale_conv1 = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1,
                                         stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
        avg_pool = self.avg_pool(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(avg_pool), 0.1, inplace=True))
        scale = torch.sigmoid(scale)
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x), 0.1, inplace=True))
        shift = (x + shift) * scale

        out = (x * s_h.expand_as(x) * s_w.expand_as(x)) * shift
        return out

# class CBB(nn.Module):
#     def __init__(self):
#         super(CBB, self).__init__()
#
#
#     def forward(self, x):
#         B, C, H, W = x.size()
#         token = x.permute(0, 2, 3, 1).view(B, -1, C)
#         mean = torch.mean(token, dim=2, keepdim=True)
#         broadcast = torch.cat([mean] * C, dim=2)
#         x = token + broadcast
#         x = x.permute(0, 2, 1).view(B, C, H, W)
#         return x

class ResNet50(nn.Module):
    def __init__(self, config, drop=0., **kwargs, ):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride = (1, 1)
            resnet50.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])

        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))

        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

        self.uncloth_dim = config.MODEL.NO_CLOTHES_DIM // 2
        self.contour_dim = config.MODEL.CONTOUR_DIM // 2
        self.cloth_dim = config.MODEL.CLOTHES_DIM // 2

        self.uncloth_net = GEN(in_feat_dim=self.uncloth_dim, out_img_dim=1, config=config)
        self.contour_net = GEN(in_feat_dim=self.contour_dim + self.cloth_dim, out_img_dim=1, config=config)
        self.cloth_net = GEN(in_feat_dim=self.cloth_dim, out_img_dim=1, config=config)

        self.local = FAA(1024, 1024)
        self.sftca = PRA(channel=512, h=24, w=12)
        self.sftca1 = PRA(channel=1024, h=24, w=12)

        dim = 1024
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )

        self.spatial_interaction1 = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, dim, kernel_size=1)
        )

        self.dwconv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction2 = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, dim, kernel_size=1)
        )

        dim1 = 512
        self.dwconv3 = nn.Sequential(
            nn.Conv2d(dim1, dim1, kernel_size=3, stride=1, padding=1, groups=dim1),
            nn.BatchNorm2d(dim1),
            nn.GELU()
        )
        self.channel_interaction3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim1, dim1 // 8, kernel_size=1),
            nn.BatchNorm2d(dim1 // 8),
            nn.GELU(),
            nn.Conv2d(dim1 // 8, dim1, kernel_size=1),
        )
        self.spatial_interaction3 = nn.Sequential(
            nn.Conv2d(dim1, dim1 // 16, kernel_size=1),
            nn.BatchNorm2d(dim1 // 16),
            nn.GELU(),
            nn.Conv2d(dim1 // 16, dim1, kernel_size=1)
        )

        self.proj_drop_1 = nn.Dropout(drop)
        self.proj_drop_2 = nn.Dropout(drop)
        self.proj_drop_3 = nn.Dropout(drop)

       # self.cbb = CBB()


    def forward(self, x):
        x = self.base(x)
        x_ori = x
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)
        f = self.bn(x)

        f_unclo = x_ori[:, 0:self.uncloth_dim, :, :]
        f_cont = x_ori[:, self.uncloth_dim:self.uncloth_dim + self.contour_dim + self.cloth_dim, :, :]
        f_clo = x_ori[:, self.uncloth_dim + self.contour_dim:self.uncloth_dim + self.contour_dim + self.cloth_dim, :, :]

        f_unclo_local = self.local(f_unclo)  # 1024
        f_cont_sftca = self.sftca1(f_cont)  # 1024
        f_clo_sftca = self.sftca(f_clo)  # 512
        #f_clo_sftca = self.cbb(f_clo_sftca)

        # print("f_unclo.shape:", f_unclo.shape)
        # print("f_cont.shape:", f_cont.shape)
        # print("f_clo.shape:", f_clo.shape)
        conv_x_1 = self.dwconv1(f_unclo)
        map1_1 = self.channel_interaction1(conv_x_1)
        map1_2 = self.spatial_interaction1(f_unclo_local)
        local_map1 = f_unclo_local * torch.sigmoid(map1_1)
        conv_map1 = conv_x_1 * torch.sigmoid(map1_2)
        f_unclo_map = local_map1 + conv_map1

        conv_x_2 = self.dwconv2(f_cont)
        map2_1 = self.channel_interaction2(conv_x_2)
        map2_2 = self.spatial_interaction2(f_cont_sftca)
        sftca_map2 = f_cont_sftca * torch.sigmoid(map2_1)
        f_cont_map_ = conv_x_2 * torch.sigmoid(map2_2)
        f_cont_map = f_cont_map_ + sftca_map2

        conv_x_3 = self.dwconv3(f_clo)
        map3_1 = self.channel_interaction3(conv_x_3)
        map3_2 = self.spatial_interaction3(f_clo_sftca)
        sftca_map3 = f_clo_sftca * torch.sigmoid(map3_1)
        f_clo_map_ = conv_x_3 * torch.sigmoid(map3_2)
        f_clo_map = f_clo_map_ + sftca_map3

        f_unclo_map = self.proj_drop_1(f_unclo_map)
        f_clo_map = self.proj_drop_2(f_clo_map)
        f_cont_map = self.proj_drop_3(f_cont_map)

        unclo_img = self.uncloth_net(f_unclo_map)
        cont_img = self.contour_net(f_cont_map)
        clo_img = self.cloth_net(f_clo_map)

        return (f, unclo_img, cont_img, clo_img)