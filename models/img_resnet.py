import torchvision
import torch
from torch import nn
from torch.nn import init
import pooling
import torch.nn.functional as F

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



        

class ResNet50(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1) 
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
        
        self.uncloth_dim = config.MODEL.NO_CLOTHES_DIM//2
        self.contour_dim = config.MODEL.CONTOUR_DIM//2
        self.cloth_dim = config.MODEL.CLOTHES_DIM//2

        self.uncloth_net = GEN(in_feat_dim = self.uncloth_dim, out_img_dim=1, config = config)
        self.contour_net = GEN(in_feat_dim = self.contour_dim + self.cloth_dim, out_img_dim=1, config = config)
        self.cloth_net = GEN(in_feat_dim = self.cloth_dim, out_img_dim=1, config = config)


        
    def forward(self, x):
        #print(x.shape)
        x = self.base(x)
        #print(x.shape)   resnet50 做了16倍的下采样
        x_ori = x

        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)
        f = self.bn(x)

        f_unclo = x_ori[:, 0:self.uncloth_dim, :, :]
        f_cont  = x_ori[:, self.uncloth_dim:self.uncloth_dim+self.contour_dim+self.cloth_dim, :, :]
        f_clo   = x_ori[:, self.uncloth_dim+self.contour_dim:self.uncloth_dim+self.contour_dim+self.cloth_dim, :, :]
        
        # print("f_unclo.shape:", f_unclo.shape)
        # print("f_cont.shape:", f_cont.shape)
        # print("f_clo.shape:", f_clo.shape)
        
        unclo_img = self.uncloth_net(f_unclo)
        cont_img  = self.contour_net(f_cont)
        clo_img   = self.cloth_net(f_clo)

        return (f, unclo_img, cont_img, clo_img)


class SFTCA(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(SFTCA, self).__init__()

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
        print(x_h.shape)
        x_w = self.avg_pool_y(x)
        print(x_w.shape)
        avg_pool = self.avg_pool(x)
        print(avg_pool.shape)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))
        print(x_cat_conv_relu.shape)
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(avg_pool), 0.1, inplace=True))
        scale = torch.sigmoid(scale)
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x), 0.1, inplace=True))
        shift = (x + shift) * scale

        print(s_h.shape)
        print(s_h.expand_as(x).shape)
        out = (x * s_h.expand_as(x) * s_w.expand_as(x)) * shift
        return out

model=SFTCA(32,128,128)
input=torch.zeros([1,32,128,128])
out=model(input)
print(out.shape)