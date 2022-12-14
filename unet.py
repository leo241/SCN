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
    def __init__(self,num_classes): # 这里的class num是指分割种类
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
        self.out=nn.Conv2d(16,num_classes,1,1,0) # 即便是最后一层，也要用卷积层的方式出来，因为出来的依然是图片的形式
        self.fc1 = nn.Linear(196*64,64)
        self.out2 = nn.Linear(64, 1) # 这里就是SCN出来地方
        self.activate = nn.Softmax(dim=1) # 将结果进行softmax，注意这里要用dim=1，
        # 也就是在第二个维度上，因为输入的x是（1，3，240，240），要对“3”个种类的分割前景进行softmax

    def forward(self,x):
        # print(x.shape) # (1,1,240,240)
        R1=self.c1(x)
        # print(R1.shape) # (1,16,240,240) # 卷积层是单纯的加厚，因为可以有无限的卷积核
        R2=self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))

        O1=self.c6(self.u1(R5,R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))
        output1 = self.out(O4) # 最终出来的分割预测
        output1 = self.activate(output1) # 经过在dim=1维度上的softmax，得到最终在几种分割前景上的概率

        # print(R5.shape)
        # 下面是SCN的部分
        R5 = DownSample(256)(R5)
        # print(R5.shape)
        R5= R5.view(R5.shape[0], -1)
        # print(R5.shape)
        fc1 = self.fc1(R5)
        out2 = self.out2(fc1) # scn层出来对于图片ID的预测

        return output1,out2



if __name__ == '__main__':
    x=torch.randn(1,1,240,240) # (batch-size, rgb_channel_size,length,height)
    net=UNet(4) # 做四分类，三个前景和一个后经
    output1,output2 = net(x) # output1是分割预测，output2是序号id预测
    print(output1.shape) # (batchsize,class_num,len,height)
    print(output2)
    loss_scn = nn.MSELoss()
    loss_entr = nn.CrossEntropyLoss()
    print(loss_scn(torch.tensor(3),output2))
    y = torch.randn(1,4,240,240)
    print(loss_entr(y,output1))