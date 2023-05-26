import torch
from torch import nn
import module.base_module as common
import torch.nn.functional as F
#from moco.builder import MoCo
from SimCLR.resblock_simclr import Resb_SimCLR
import torch.nn.init as init


def make_model(args):
    return BlindSR(args)
def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class SFT_Layer(nn.Module):
    def __init__(self, nf=64, para=10):
        super(SFT_Layer, self).__init__()
        self.mul_conv1 = nn.Conv2d(para + nf, 32, kernel_size=3, stride=1, padding=1)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

        self.add_conv1 = nn.Conv2d(para + nf, 32, kernel_size=3, stride=1, padding=1)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_maps, para_maps):
        cat_input = torch.cat((feature_maps, para_maps), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.add_leaky(self.add_conv1(cat_input)))
        return feature_maps * mul + add

class ResidualDenseBlock_5C(nn.Module):
    '''  Residual Dense Block '''
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB_SFT(nn.Module):
    ''' Residual in Residual Dense Block with SFT layer '''

    def __init__(self, nf=64, gc=32, para=15):
        super(RRDB_SFT, self).__init__()
        self.SFT = SFT_Layer(nf=nf, para=para)
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, fea, ker):
        out = self.SFT(fea, ker)
        out = self.RDB1(out)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + fea

class SFT_Residual_Block(nn.Module):
    def __init__(self, nf=64, para=10):
        super(SFT_Residual_Block, self).__init__()
        self.sft1 = SFT_Layer(nf=nf, para=para)
        self.sft2 = SFT_Layer(nf=nf, para=para)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, feature_maps, para_maps):
        fea1 = F.relu(self.sft1(feature_maps, para_maps))
        fea2 = F.relu(self.sft2(self.conv1(fea1), para_maps))
        fea3 = self.conv2(fea2)
        return torch.add(feature_maps, fea3)


class SFTMD(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=10, scale=4, input_para=15, min=0.0, max=1.0):
        super(SFTMD, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = nb

        self.conv1 = nn.Conv2d(in_nc, 64, 3, stride=1, padding=1)
        self.relu_conv1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.relu_conv2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        # sft_branch is not used.
        sft_branch = []
        for i in range(nb):
            sft_branch.append(RRDB_SFT())
        self.sft_branch = nn.Sequential(*sft_branch)

        for i in range(nb):
            self.add_module('SFT-residual' + str(i + 1), RRDB_SFT(nf=nf, gc=32, para=self.para))

        self.sft = SFT_Layer(nf=64, para=input_para)
        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

        # compress
        self.compress = nn.Sequential(
            nn.Linear(256, self.para, bias=False),
            nn.LeakyReLU(0.1, True)
        )

        if scale == 4:  # x4
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64 * scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64 * scale, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:  # x2, x3
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64 * scale ** 2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=out_nc, kernel_size=9, stride=1, padding=4, bias=True)

    def forward(self, input, ker_code):
        B, C, H, W = input.size()  # I_LR batch
        ker_code = self.compress(ker_code)
        B_h, C_h = ker_code.size()  # Batch, Len=10
        ker_code_exp = ker_code.view((B_h, C_h, 1, 1)).expand((B_h, C_h, H, W))  # kernel_map stretch

        fea_bef = self.conv3(self.relu_conv2(self.conv2(self.relu_conv1(self.conv1(input)))))
        fea_in = fea_bef
        for i in range(self.num_blocks):
            fea_in = self.__getattr__('SFT-residual' + str(i + 1))(fea_in, ker_code_exp)
        fea_mid = fea_in
        # fea_in = self.sft_branch((fea_in, ker_code_exp))
        fea_add = torch.add(fea_mid, fea_bef)
        fea = self.upscale(self.conv_mid(self.sft(fea_add, ker_code_exp)))
        out = self.conv_output(fea)

        return out

class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(64, 10, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(10, 10, bias=False)
        )
        # self.conv = common.default_conv(channels_in, channels_out, 1)
        self.add_conv1 = nn.Conv2d(64+10, 32, kernel_size=3, stride=1, padding=1)
        self.add_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.ca = CA_layer(channels_in, channels_out, reduction)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()

        # branch 1
        kernel = self.kernel(x[1])
        kernel_exp = kernel.view((b, 10, 1, 1)).expand((b, 10, h, w))
        out = torch.cat((x[0], kernel_exp), dim=1)
        out = self.add_conv2(self.relu(self.add_conv1(out)))
        # out = self.relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
        # out = self.conv(out.view(b, -1, h, w))

        # branch 2
        out = out + self.ca(x)

        return out


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        att = self.conv_du(x[1][:, :, None, None])

        return x[0] * att


class DAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(DAB, self).__init__()

        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu =  nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''

        out = self.relu(self.da_conv1(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2([out, x[1]]))
        out = self.conv2(out) + x[0]

        return out


class DAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super(DAG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            DAB(conv, n_feat, kernel_size, reduction) \
            for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        res = x[0]
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1]])
        res = self.body[-1](res)
        res = res + x[0]

        return res


class DASR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DASR, self).__init__()

        self.n_groups = 5
        n_blocks = 5
        n_feats = 64
        kernel_size = 3
        reduction = 8
        scale = int(args.scale)


        # head module
        modules_head = [conv(3, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # compress
        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

        # body
        modules_body = [
            DAG(common.default_conv, n_feats, kernel_size, reduction, n_blocks) \
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                        conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, k_v):
        k_v = self.compress(k_v)

        # head
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_groups):
            res = self.body[i]([res, k_v])
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out


class BlindSR(nn.Module):
    def __init__(self, args):
        super(BlindSR, self).__init__()

        # Generator
        self.GE = DASR(args)
        # self.G = SFTMD()

        # Encoder
        # self.E = MoCo(base_encoder=Encoder)
        self.E = Resb_SimCLR(encoder=Encoder())

    def forward(self, lr_d_i, lr_d_j):
        if self.training:
            x_query = lr_d_i                          # b, c, h, w
            x_key = lr_d_j                           # b, c, h, w

            # degradation-aware represenetion learning
            h_i, h_j, mlp_i, mlp_j = self.E(x_query, x_key)

            # degradation-aware SR
            #h_lr, _, _, _ = self.E(lr, lr)
            sr_i = self.GE(lr_d_i, h_i)

            return sr_i, mlp_i, mlp_j
        else:
            # degradation-aware represenetion learning
            h_lr_i, _, _, _ = self.E(lr_d_i, lr_d_i)

            # degradation-aware SR
            sr = self.GE(lr_d_i, h_lr_i)

            return sr