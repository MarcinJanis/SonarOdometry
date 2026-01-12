import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_ch = 64, out_ch = 64, stride = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride = stride, padding = 1)
        self.norm1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride = 1, padding = 1)
        self.norm2 = nn.BatchNorm2d(out_ch)


        if stride != 1 or in_ch != out_ch:
            self.downsample = True
            self.downsample_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride = stride)
        else:
            self.downsample = False

    def forward(self, x):

        x_res = x 

        x = self.conv1(x)
        x = F.relu(self.norm1(x))

        x = self.conv2(x)
        x = self.norm2(x)

        if self.downsample:
            x_res = self.downsample_conv(x_res)
        
        return F.relu(x + x_res)



class Encoder(nn.Module):
    def __init__(self, in_ch = 1, out_ch = 128, dim = 32, dropout=0.5):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, dim, kernel_size=7, stride=2, padding=3)

        self.conv2 = nn.Conv2d(dim*2, out_ch, kernel_size=3, stride=1, padding=1)

        self.ResBlock1 = nn.Sequential(ResidualBlock(in_ch=dim, out_ch=dim, stride=1),
                                       ResidualBlock(in_ch=dim, out_ch=dim, stride=1))

        self.ResBlock2 = nn.Sequential(ResidualBlock(in_ch=dim, out_ch=dim*2, stride=2),
                                       ResidualBlock(in_ch=dim*2, out_ch=dim*2, stride=1))
        
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        b, n, c1, h1, w1 = x.shape
        x = x.view(b*n, c1, h1, w1)

        x = F.relu(self.conv1(x))
        x = self.ResBlock1(x)
        x = self.ResBlock2(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))

        _, c2, h2, w2 = x.shape

        return x.view(b, n, c2, h2, w2)
