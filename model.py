
import torch
import torch.nn as nn
from resnet import resnet50
import torch.nn.functional as F

class conv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x

class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)

class dilated_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.c1 = nn.Sequential(conv2d(in_c, out_c, kernel_size=1, padding=0), channel_attention(out_c))
        self.c2 = nn.Sequential(conv2d(in_c, out_c, kernel_size=(3, 3), padding=6, dilation=6), channel_attention(out_c))
        self.c3 = nn.Sequential(conv2d(in_c, out_c, kernel_size=(3, 3), padding=12, dilation=12), channel_attention(out_c))
        self.c4 = nn.Sequential(conv2d(in_c, out_c, kernel_size=(3, 3), padding=18, dilation=18), channel_attention(out_c))
        self.c5 = conv2d(out_c*4, out_c, kernel_size=3, padding=1, act=False)
        self.c6 = conv2d(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.sa = spatial_attention()

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        xc = torch.cat([x1, x2, x3, x4], axis=1)
        xc = self.c5(xc)
        xs = self.c6(x)
        x = self.relu(xc+xs)
        x = self.sa(x)
        return x

class label_attention(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        """ Channel Attention """
        self.c1 = nn.Sequential(
            nn.Conv2d(in_c[1], in_c[0], kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_c[0], in_c[0], kernel_size=1, padding=0, bias=False)
        )

    def forward(self, feats, label):
        """ Channel Attention """
        b, c = label.shape
        label = label.reshape(b, c, 1, 1)
        ch_attn = self.c1(label)
        ch_map = torch.sigmoid(ch_attn)
        feats = feats * ch_map

        ch_attn = ch_attn.reshape(ch_attn.shape[0], ch_attn.shape[1])
        return ch_attn, feats

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, scale=2):
        super().__init__()
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.c1 = conv2d(in_c+out_c, out_c, kernel_size=1, padding=0)
        self.c2 = conv2d(out_c, out_c, act=False)
        self.c3 = conv2d(out_c, out_c, act=False)
        self.c4 = conv2d(out_c, out_c, kernel_size=1, padding=0, act=False)
        self.ca = channel_attention(out_c)
        self.sa = spatial_attention()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)
        x = self.c1(x)

        s1 = x
        x = self.c2(x)
        x = self.relu(x+s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x+s2+s1)

        s3 = x
        x = self.c4(x)
        x = self.relu(x+s3+s2+s1)

        x = self.ca(x)
        x = self.sa(x)
        return x

class output_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.up(x)
        x = self.c1(x)
        return x

class text_classifier(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_c, in_c//8, bias=False), nn.ReLU(),
            nn.Linear(in_c//8, out_c[0], bias=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_c, in_c//8, bias=False), nn.ReLU(),
            nn.Linear(in_c//8, out_c[1], bias=False)
        )

    def forward(self, feats):
        pool = self.avg_pool(feats).view(feats.shape[0], feats.shape[1])
        num_polyps = self.fc1(pool)
        polyp_sizes = self.fc2(pool)
        return num_polyps, polyp_sizes

class embedding_feature_fusion(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Conv2d((in_c[0]+in_c[1])*in_c[2], out_c, 1, bias=False), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 1, bias=False), nn.ReLU()
        )

    def forward(self, num_polyps, polyp_sizes, label):
        num_polyps_prob = torch.softmax(num_polyps, axis=1)
        polyp_sizes_prob = torch.softmax(polyp_sizes, axis=1)
        prob = torch.cat([num_polyps_prob, polyp_sizes_prob], axis=1)
        prob = prob.view(prob.shape[0], prob.shape[1], 1)
        x = label * prob
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.fc(x)
        x = x.view(x.shape[0], -1)
        return x

class multiscale_feature_aggregation(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.c11 = conv2d(in_c[0], out_c, kernel_size=1, padding=0)
        self.c12 = conv2d(in_c[1], out_c, kernel_size=1, padding=0)
        self.c13 = conv2d(in_c[2], out_c, kernel_size=1, padding=0)
        self.c14 = conv2d(out_c*3, out_c, kernel_size=1, padding=0)

        self.c2 = conv2d(out_c, out_c, act=False)
        self.c3 = conv2d(out_c, out_c, act=False)

    def forward(self, x1, x2, x3):
        x1 = self.up_4x4(x1)
        x2 = self.up_2x2(x2)

        x1 = self.c11(x1)
        x2 = self.c12(x2)
        x3 = self.c13(x3)
        x = torch.cat([x1, x2, x3], axis=1)
        x = self.c14(x)

        s1 = x
        x = self.c2(x)
        x = self.relu(x+s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x+s2+s1)

        return x

class TGAPolypSeg(nn.Module):
    def __init__(self):
        super().__init__()

        """ Backbone: ResNet50 """
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.text_classifier = text_classifier(1024, [2, 3])
        self.label_fc = embedding_feature_fusion([2, 3, 300], 128)

        """ Dilated Conv """
        self.s1 = dilated_conv(64, 128)
        self.s2 = dilated_conv(256, 128)
        self.s3 = dilated_conv(512, 128)
        self.s4 = dilated_conv(1024, 128)

        """ Decoder """
        self.d1 = decoder_block(128, 128, scale=2)
        self.a1 = label_attention([128, 128])

        self.d2 = decoder_block(128, 128, scale=2)
        self.a2 = label_attention([128, 128])

        self.d3 = decoder_block(128, 128, scale=2)
        self.a3 = label_attention([128, 128])

        self.ag = multiscale_feature_aggregation([128, 128, 128], 128)

        self.y1 = output_block(128, 1)

    def forward(self, image, label):
        """ Backbone: ResNet50 """
        x0 = image
        x1 = self.layer0(x0)    ## [-1, 64, h/2, w/2]
        x2 = self.layer1(x1)    ## [-1, 256, h/4, w/4]
        x3 = self.layer2(x2)    ## [-1, 512, h/8, w/8]
        x4 = self.layer3(x3)    ## [-1, 1024, h/16, w/16]
        # print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)

        num_polyps, polyp_sizes = self.text_classifier(x4)
        f0 = self.label_fc(num_polyps, polyp_sizes, label)

        """ Dilated Conv """
        s1 = self.s1(x1)
        s2 = self.s2(x2)
        s3 = self.s3(x3)
        s4 = self.s4(x4)
        # print(s1.shape, s2.shape, s3.shape, s4.shape)

        """ Decoder """
        d1 = self.d1(s4, s3)
        f1, a1 = self.a1(d1, f0)

        d2 = self.d2(a1, s2)
        f = f0 + f1
        f2, a2 = self.a2(d2, f)

        d3 = self.d3(a2, s1)
        f = f0 + f1 + f2
        f3, a3 = self.a3(d3, f)

        ag = self.ag(a1, a2, a3)
        y1 = self.y1(ag)

        return y1, num_polyps, polyp_sizes

def prepare_input(res):
    x1 = torch.FloatTensor(1, 3, 256, 256).cuda()
    x2 = torch.FloatTensor(1, 5, 300).cuda()
    return dict(x = [x1, x2])

if __name__ == "__main__":
    # x = torch.randn((4, 3, 256, 256))
    # l = torch.randn((4, 5, 300))
    # model = TGAPolypSeg()
    #
    # from ptflops import get_model_complexity_info
    # flops, params = get_model_complexity_info(model, input_res=(3, 256, 256), input_constructor=prepare_input, as_strings=True, print_per_layer_stat=False)
    # print('      - Flops:  ' + flops)
    # print('      - Params: ' + params)

    from ptflops import get_model_complexity_info

    model = TGAPolypSeg().cuda()
    flops, params = get_model_complexity_info(model, input_res=(512, 512),
                                              input_constructor=prepare_input,
                                              as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)

    # from profile import profile
    # input_size = [(1, 3, 512, 512), (1, 1, 512, 512)]
    # # custom_ops = { '<your_layer_name' : '<your_custom_count_layer_function>', ... }
    # num_ops, num_params = profile(model, input_size)
    # print(num_ops, num_params)
