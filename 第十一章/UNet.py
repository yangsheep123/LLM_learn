import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        self.model = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        return self.model(x)


class UNet(nn.Module):
    def __init__(self,in_channels,out_channels):
        self.conv1 = DoubleConv(in_channels,64)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = DoubleConv(64,128)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3 = DoubleConv(128,256)
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv4 = DoubleConv(256,512)
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv5 = DoubleConv(512,1024)
        self.pool5 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.up6 = nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=2,stride=2)
        self.conv6 = DoubleConv(1024,512) # up6和conv5进行拼接，512+512=1024
        self.up7 = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=2,stride=2)
        self.conv7 = DoubleConv(512,256)
        self.up8 = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2)
        self.conv8 = DoubleConv(256,128)
        self.up9 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2)
        self.conv9 = DoubleConv(128,64)
        self.conv10 = nn.Conv2d(64,out_channels,kernel_size=1)

    def forward(self,x):
        # 编码
        c1 = self.conv1(x)          #c1:[B,64,H,W]
        p1 = self.pool1(c1)         #p1:[B,64,H/2,W/2]
        c2 = self.conv2(p1)         #c2:[B,128,H/2,W/2]
        p2 = self.pool2(c2)         #p2:[B,128,H/4,W/4]
        c3 = self.conv3(p2)         #c3:[B,256,H/4,W/4]
        p3 = self.pool3(c3)         #p3:[B,256,H/8,W/8]
        c4 = self.conv4(p3)         #c4:[B,512,H/8,W/8]
        p4 = self.pool3(c4)         #p4:[B,512,H/16,W/16]
        c5 = self.conv5(p4)         #c5:[B,1024,H/16,W/16]
        p5 = self.pool5(c5)         #p5:[B,1024,H/32,W/32]

        # 解码
        up6 = self.up6(p5)          #up6:[B,512,H/16,W/16]
        up6 = torch.cat([up6,c4],dim=1)#[B,1024,H/16,W/16]
        c6 = self.conv6(up6)        #c6:[B,512,H/16,W/16]
        up7 = self.up7(c6)          #up7:[B,256,H/8,W/8]
        up7 = torch.cat([up7,c3],dim=1)
        c7 = self.conv7(up7)        #c7:[B,256,H/8,W/8]
        up8 = self.up8(c7)          #up8:[B,128,H/4,W/4]
        up8 = torch.cat([up8,c2],dim=1)
        c8 = self.up8(up8)          #c8:[B,128,H/4,W/4]

        # 解码，第 4 级
        u9 = self.up9(c8)       # [B,64, H, W]
        u9 = torch.cat([c1, u9], dim=1)  # [B,128,H,W]
        c9 = self.conv9(u9)     # [B,64, H, W]

        out = self.conv10(c9)   # [B,out_channels,H,W]
        return out






