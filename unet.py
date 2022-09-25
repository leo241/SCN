import torch
from torch import nn
from torch.nn import functional as F

class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    def forward(self,x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            nn.MaxPool2d(kernel_size=2)
        )
    def forward(self,x):
        return self.layer(x)

class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample, self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)
    def forward(self,x,feature_map):
        up=F.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)




class UNet(nn.Module):
    def __init__(self,num_classes):
        super(UNet, self).__init__()
        self.c1=Conv_Block(1,16) # first para means rgb-channel size,for gray figure choose 1
        self.d1=DownSample(16)
        self.c2=Conv_Block(16,32)
        self.d2=DownSample(32)
        self.c3=Conv_Block(32,64)
        self.d3=DownSample(64)
        self.c4=Conv_Block(64,128)
        self.d4=DownSample(128)
        self.c5=Conv_Block(128,256)
        self.u1=UpSample(256)
        self.c6=Conv_Block(256,128)
        self.u2 = UpSample(128)
        self.c7 = Conv_Block(128, 64)
        self.u3 = UpSample(64)
        self.c8 = Conv_Block(64, 32)
        self.u4 = UpSample(32)
        self.c9 = Conv_Block(32, 16)
        self.out=nn.Conv2d(16,num_classes,1,1,0)
        self.fc1 = nn.Linear(196*64,64)
        self.out2 = nn.Linear(64, 1)

    def forward(self,x):
        R1=self.c1(x)
        R2=self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))

        O1=self.c6(self.u1(R5,R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        # print(R5.shape)
        R5 = DownSample(256)(R5)
        # print(R5.shape)
        R5= R5.view(R5.shape[0], -1)
        # print(R5.shape)
        fc1 = self.fc1(R5)
        out2 = self.out2(fc1)

        return self.out(O4),out2

if __name__ == '__main__':
    x=torch.randn(1,1,240,240) # (batch-size, rgb_channel_size,length,height)
    net=UNet(3) # 做三分类
    output1,output2 = net(x)
    print(output1.shape) # (batchsize,class_num,len,height)
    print(output2)
    loss = nn.MSELoss()
    print(loss(torch.tensor(3),output2))