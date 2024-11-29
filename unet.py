import torch
import torch.nn as nn
import torch.nn.functional as F
from priors import *
from torch.nn.modules.utils import _pair
from torch.nn import init


class DropConnectConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 needs_drop=True,
                 drop_prob=0.2,
                 bias=False):
        bias = False
        super(DropConnectConv, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.use_bias = bias
        self.needs_drop = needs_drop
        self.drop_prob = drop_prob
        self.exec_time = 0
        self.floatTensor = (torch.FloatTensor if not torch.cuda.is_available() else
                            torch.cuda.FloatTensor)
        self.weight = nn.Parameter(
            self.floatTensor(out_channels, in_channels, *self.kernel_size),
            requires_grad=True)
        if bias:
            self.bias = nn.Parameter(
                self.floatTensor(out_channels),
                requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()
        self.Droplayer = nn.Dropout(p=self.drop_prob)

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, x):
        if self.needs_drop:
            self.Droplayer.training = True
        kernel = self.Droplayer(self.weight)
        pq = torch.sum(kernel ** 2)
        return F.conv2d(x, kernel,
                        self.bias, self.stride, self.padding,
                        self.dilation, self.groups), pq

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None, droprate=None, needs_drop=True,
                 prior_instance=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class DoubleConvBeta(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None, droprate=None, needs_drop=True,
                 prior_instance=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=droprate),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=droprate),
        )
    
    def forward(self, x):
        return self.double_conv(x)


class DoubleConvBBB(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, droprate=None, needs_drop=True,
                 prior_instance=None):
        super().__init__()
        self.prior_instance = prior_instance
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = BayesConv_Normalq(in_channels, mid_channels, kernel_size=3, padding=1,
                                       prior_class=self.prior_instance)
        self.relu1 = nn.Sequential(nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))
        self.conv2 = BayesConv_Normalq(mid_channels, out_channels, kernel_size=3, padding=1,
                                       prior_class=self.prior_instance)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, sample=True):
        tlqw = 0
        tlpw = 0
        x, lqw, lpw = self.conv1(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        x = self.relu1(x)
        x, lqw, lpw = self.conv2(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        x = self.relu2(x)
        return x, tlqw, tlpw

class DoubleConvDrop(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, droprate=0.5, needs_drop=True,
                 prior_instance=None):
        super().__init__()
        self.needs_drop = needs_drop
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=None)
        self.relu1 = nn.Sequential(nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))
        # self.relu1 = nn.Sequential(nn.ReLU(inplace=True))
        self.drop1 = nn.Dropout2d(p=droprate)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=None)
        # self.relu2 = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.relu2 = nn.Sequential(nn.ReLU(inplace=True))
        self.drop2 = nn.Dropout2d(p=droprate)

    def forward(self, x):
        if self.needs_drop:
            self.drop1.training = True
            self.drop2.training = True
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        regu = torch.sum(self.conv1.weight ** 2) + torch.sum(self.conv2.weight ** 2)

        return x, regu

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, ConvBlock=DoubleConv, droprate=0.5, needs_drop=True,
                 prior_instance=None):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, droprate=droprate, needs_drop=needs_drop,
                              prior_instance=prior_instance)
        # self.conv = DoubleConv(in_channels, out_channels, droprate=droprate, needs_drop=needs_drop, prior_instance = prior_instance)
        if ConvBlock == DoubleConvBBB:
            self.bbb = True
        else:
            self.bbb = False
        if ConvBlock == DoubleConvDrop:
            self.dcp = True
        else:
            self.dcp = False

    def forward(self, x):
        if self.bbb:
            x, tlqw, tlpw = self.conv(self.down(x))
            return x, tlqw, tlpw
        elif self.dcp:
            x, qp = self.conv(self.down(x))
            return x, qp
        else:
            return self.conv(self.down(x))

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, ConvBlock=DoubleConv, droprate=0.5, needs_drop=True,
                 prior_instance=None):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels, droprate=droprate, needs_drop=needs_drop,
                              prior_instance=prior_instance)
        if ConvBlock == DoubleConvBBB:
            self.bbb = True
        else:
            self.bbb = False
        if ConvBlock == DoubleConvDrop:
            self.dcp = True
        else:
            self.dcp = False

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        if self.bbb:
            x, tlqw, tlpw = self.conv(x)
            return x, tlqw, tlpw
        elif self.dcp:
            x, qp = self.conv(x)
            return x, qp
        else:
            return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        # class 4
        out_channels = 4
        super(OutConv, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=None)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=None)
        #self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

class OutConv_Beta(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_Beta, self).__init__()
        # class 4
        out_channels = 4
        self.fc1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=None)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=None)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        #return torch.exp(x) + 1
        return torch.exp(torch.abs(x))
        #return self.sigmoid(x)


class OutConv_MC(nn.Module):
    def __init__(self, in_channels, out_channels, droprate=0.5, needs_drop=True):
        super(OutConv_MC, self).__init__()
        # class 4
        out_channels = 4
        self.needs_drop = needs_drop
        self.fc1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=None)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d(p=droprate)
        self.fc2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=None)
        #self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        if self.needs_drop:
            self.drop1.training = True
        x = self.relu1(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)
        regu = torch.sum(self.fc1.weight ** 2) + torch.sum(self.fc2.weight ** 2)
        #return self.sigmoid(x), regu
        return self.softmax(x), regu

class OutConv_BBB(nn.Module):
    def __init__(self, in_channels, out_channels, prior_instance):
        super(OutConv_BBB, self).__init__()
        # class 4
        out_channels = 4
        self.prior_instance = prior_instance
        self.fc1 = BayesLinear_Normalq(in_channels, in_channels, self.prior_instance)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = BayesLinear_Normalq(in_channels, out_channels, self.prior_instance)
        #self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x, sample):
        b, c, h, w = x.shape
        tlqw = 0
        tlpw = 0
        x = x.permute(0, 2, 3, 1)
        x = x.reshape([b * h * w, c])
        x, lqw, lpw = self.fc1(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        x = self.relu1(x)
        x, lqw, lpw = self.fc2(x, sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        x = x.reshape([b, h, w, 4])
        x = x.permute(0, 3, 1, 2)
        #return self.sigmoid(x), tlqw, tlpw
        return self.softmax(x), tlqw, tlpw


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, mode='RAW', prior_instance=None, droprate=0.5,
                 droprateBack=0.1, model_c=64):
        super(UNet, self).__init__()
        c = model_c

        self.mode = mode
        if mode == 'MCDrop' or mode == 'NaiveDrop':
            ConvBlock = DoubleConvDrop
        elif mode == 'BBB':
            ConvBlock = DoubleConvBBB
        elif mode == 'dir':
            ConvBlock = DoubleConvBeta
            #print('ha')
        else:
            ConvBlock = DoubleConv
        if mode == 'MCDrop':
            needs_drop = True
        else:
            needs_drop = False

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = ConvBlock(n_channels, c, droprate=droprateBack, needs_drop=needs_drop,
                             prior_instance=prior_instance)
        self.down1 = Down(c, int(c * 2), ConvBlock, droprate=droprateBack, needs_drop=needs_drop,
                          prior_instance=prior_instance)
        factor = 2
        self.down2 = Down(int(c * 2), int(c * 4) // factor, ConvBlock, droprate=droprateBack, needs_drop=needs_drop,
                          prior_instance=prior_instance)
        self.up3 = Up(int(c * 4), int(c * 2) // factor, ConvBlock, droprate=droprateBack, needs_drop=needs_drop,
                      prior_instance=prior_instance)
        self.up4 = Up(int(c * 2), c, ConvBlock, droprate=droprateBack, needs_drop=needs_drop,
                      prior_instance=prior_instance)
        if mode == 'RAW':
            self.outc = OutConv(c, n_classes)
        elif mode == 'dir':
            self.outc = OutConv_Beta(c, n_classes)
        elif mode == 'MCDrop' or mode == 'NaiveDrop':
            self.outc = OutConv_MC(c, n_classes, droprate=droprate, needs_drop=needs_drop)
        elif mode == 'BBB':
            self.outc = OutConv_BBB(c, n_classes, prior_instance)
        else:
            print('error')

    def forward(self, x, sample=True):
        if self.mode == 'BBB':
            x1, tlqw1, tlpw1 = self.inc(x)
            x2, tlqw2, tlpw2 = self.down1(x1)
            x3, tlqw3, tlpw3 = self.down2(x2)
            x = x3
            x, tlqw8, tlpw8 = self.up3(x, x2)
            x, tlqw9, tlpw9 = self.up4(x, x1)
            logits, tlqw10, tlpw10 = self.outc(x, sample)
            Ttlqw = tlqw1 + tlqw2 + tlqw3 + tlqw8 + tlqw9 + tlqw10
            Ttlpw = tlpw1 + tlpw2 + tlpw3 + tlpw8 + tlpw9 + tlpw10
            return logits, Ttlqw, Ttlpw
        elif self.mode == 'MCDrop' or self.mode == 'NaiveDrop':
            x1, tpq1 = self.inc(x)
            x2, tpq2 = self.down1(x1)
            x3, tpq3 = self.down2(x2)
            x = x3
            x, tpq8 = self.up3(x, x2)
            x, tpq9 = self.up4(x, x1)
            logits, tpq10 = self.outc(x)
            Ttpq = tpq1 + tpq2 + tpq3 + tpq8 + tpq9 + tpq10
            if self.mode == 'NaiveDrop':
                Ttpq = Ttpq * 0
            return logits, Ttpq
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x = x3
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            return logits

class UNets(nn.Module):
    def __init__(self, n_Num = 10, model_c=64):
        super(UNets, self).__init__()
        self.mode = 'Ensemble'
        self.n_Num = n_Num
        self.models = nn.ModuleList()
        for i in range(self.n_Num):
            self.models.append(UNet(model_c = model_c))

    def forward(self, x, model_id):
        logits = self.models[model_id](x)
        return logits