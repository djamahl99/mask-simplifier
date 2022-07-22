import torch
import torch.nn as nn
from generate_dataset import MAX_SEQ_LEN

from torchvision import models

def convrelu(in_channels, out_channels, kernel, padding, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, bias=bias),
        nn.ReLU(inplace=True),
    )

class PolygonPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # filters = [256, 512, 1024, 512, 256]
        filters = [128, 128, 256, 64, 32]
        # filters = [32, 64, 128, 64, 32] # when using layer before addition can have different sizes
        # filters = [64, 64, 64, 64, 64] # filter sizes have to be the same for addition

        bias = False

        self.conv1 = convrelu(1, filters[0], 3, 1, bias=bias)
        self.l1 = convrelu(filters[0], 1, 1, 0)
        self.batch_norm1 = nn.BatchNorm2d(filters[0])

        self.conv2 = convrelu(filters[0], filters[1], 3, 1, bias=bias)
        self.l2 = convrelu(filters[1], 1, 1, 0)
        self.batch_norm2 = nn.BatchNorm2d(filters[1])
        
        self.conv3 = convrelu(filters[1], filters[2], 3, 1, bias=bias)
        self.l3 = convrelu(filters[2], 1, 1, 0)
        self.batch_norm3 = nn.BatchNorm2d(filters[2])
        
        self.conv4 = convrelu(filters[2], filters[3], 3, 1, bias=bias)
        self.l4 = convrelu(filters[3], 1, 1, 0)
        self.batch_norm4 = nn.BatchNorm2d(filters[3])
        
        self.conv5 = convrelu(filters[3], filters[4], 3, 1, bias=bias)
        self.batch_norm5 = nn.BatchNorm2d(filters[4])

        self.out = nn.Sequential(
            nn.Conv2d(filters[4], 1, 1, 1),
            nn.Sigmoid()
        )

        # self.angle = nn.Sequential(
        #     nn.Conv2d(filters[4], 64, 3, 1, 1),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 1, 3, 1, 1),
        #     nn.ReLU(True),
        # )

        self.len_pooling = nn.Sequential(
            nn.AvgPool2d(3, 2), # 112
            nn.AvgPool2d(3, 2), # 56
            nn.AvgPool2d(3, 2), # 28
            # nn.AvgPool2d(3, 2), # 14
            # conv3x3(1, 16, 2),
            # conv3x3(16, 16, 2),
            # conv3x3(16, 16, 2),
            # conv3x3(16, 1, 2)
        )

        self.len_head = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, MAX_SEQ_LEN),
            nn.Sigmoid()
        )

        self.maxpool = nn.MaxPool2d(5, 2, return_indices=True)
        self.maxunpool = nn.MaxUnpool2d(5, 2)

    def forward(self, input):
        x1 = self.conv1(input)
        x1 = x1 + input
        x1 = self.batch_norm1(x1)

        x2 = self.conv2(x1)
        x2 = x2 + self.l1(x1)
        # x2 = x2 + x1
        x2 = self.batch_norm2(x2)

        x3 = self.conv3(x2)
        x3 = x3 + self.l2(x2)
        # x3 = x3 + x2
        x3 = self.batch_norm3(x3)

        x4 = self.conv4(x3)
        x4 = x4 + self.l3(x3)
        # x4 = x4 + x3
        x4 = self.batch_norm4(x4)

        x5 = self.conv5(x4)
        x5 = x5 + self.l4(x4)
        # x5 = x5 + x4
        x5 = self.batch_norm5(x5)

        out = self.out(x5)
        # angle = self.angle(x5)

        out, indices = self.maxpool(out)
        out = self.maxunpool(out, indices, output_size=input.size())

        pool = self.len_pooling(out)
        pool = torch.flatten(pool, 1)
        length = self.len_head(pool)

        return length, out

class ResNetUNet(nn.Module):
    def __init__(self, n_class, backbone='resnet18'):
        super().__init__()

        if backbone == 'resnet18':
            self.base_model = models.resnet18(pretrained=True)
        elif backbone == 'resnet34':
            self.base_model = models.resnet34(pretrained=True)
        else:
            self.base_model = models.resnet50(pretrained=True)

        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.upsample = nn.MaxUnpool2d(3, 1)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        # self.conv_last = nn.Conv2d(64, n_class, 1)
        self.conv_last = nn.Sequential(
            nn.Conv2d(64, n_class, 1),
            nn.Sigmoid()
        )

        # added for length head #################
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.avgpool = nn.Sequential(
            nn.AvgPool2d(3, 2), # 112
            nn.AvgPool2d(3, 2), # 56
            nn.AvgPool2d(3, 2), # 28
        )

        self.length_head = nn.Sequential(
            nn.LazyLinear(128),
            # nn.ReLU(),            
            # nn.Linear(1024, 512),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128,MAX_SEQ_LEN),
            nn.Sigmoid() # added as using BCE
        )

        self.pool = nn.MaxPool2d(5, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(5, 2)

        self.rgb_conv = convrelu(1, 3, 3, 1)

        self.last = convrelu(2, 1, 3, 1)

    def forward(self, input):
        # grayscale to rgb learnable?
        input = self.rgb_conv(input) 
        # end grayscale

        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        # x = self.upsample(layer4, output_size=layer3.size())
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        # x = self.upsample(x, output_size=layer2.size())

        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        # x = self.upsample(x, output_size=layer1.size())

        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        # x = self.upsample(x, output_size=layer0.size())

        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        # x = self.upsample(x, output_size=x_original.size())

        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)
        # out = self.soft(out)
        out, indices = self.pool(out)
        out = self.unpool(out, indices, output_size=x.size())
        # out = torch.cat([out, out_o], dim=1)
        
        # out = self.last(out)

        # my crap #################
        l4_flat = self.avgpool(out) # or layer4
        l4_flat = torch.flatten(l4_flat, 1)
        length = self.length_head(l4_flat)

        return length, out