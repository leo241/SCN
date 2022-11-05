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




class En_Decoder(nn.Module):
    def __init__(self,num_classes): # 这里的class num是指分割种类
        super(En_Decoder, self).__init__()
        self.c1=Conv_Block(4,16) # first para means rgb-channel size,for gray figure choose 1
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
        self.fc1 = nn.Linear(256*15*15,100)
        self.fc2 = nn.Linear(100,64)
        self.fc3 = nn.Linear(64,100)
        self.fc4= nn.Linear(100,256*15*15)
        self.out2 = nn.Linear(64, 1) # 这里就是SCN出来地方
        self.activate = nn.Softmax(dim=1) # 将结果进行softmax，注意这里要用dim=1，
        # 也就是在第二个维度上，因为输入的x是（1，3，240，240），要对“3”个种类的分割前景进行softmax
        self.c10 = Conv_Block(1,2)
        self.c11 = Conv_Block(2,4)

    def forward(self,x):
        R1=self.c1(x)
        R2=self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))

        # 开始auto encoding的全连接部分
        R5 = R5.view(R5.shape[0], -1) # flatten
        fc1 = self.fc1(R5) # 编码
        fc2 = self.fc2(fc1) # 编码
        fc3 = self.fc3(fc2) # 解码
        fc4 = self.fc4(fc3).reshape(-1,1,240,240) #解码并reshape,这里要根据batch size进行修改

        u1 = self.c10(fc4)
        u2 = self.c11(u1)
        output = self.activate(u2)

        return output

if __name__ == '__main__':
    x=torch.randn(16,4,240,240) # (batch-size, rgb_channel_size,length,height)
    net=En_Decoder(4) # 做四分类，三个前景和一个后经
    output1 = net(x) # output1是分割预测，output2是序号id预测
    print(output1.shape) # (batchsize,class_num,len,height)