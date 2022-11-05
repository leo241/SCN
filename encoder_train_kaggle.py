from PIL import Image
import numpy as np
import os
os.chdir('/kaggle/input/encoder') # 在kaggle上设置工作路径
from auto_encoder import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from random import randint

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")# 使用gpu加速



def get_oldest_model_name():
    create_folder('encoder')
    model_name_list = os.listdir('encoder')
    length = len(model_name_list)
    for i in range(length):
        model_name_list[i] = int(model_name_list[i].replace('.pth', ''))
    model_name = max(model_name_list)  # 最大的，也是最近训练的、迭代次数最多的模型
    return 'encoder/' + str(model_name) + '.pth'

def mask_one_hot(img_arr):
    myo = 125
    lv = 220
    rv = 255
    bg_max = min(myo, lv, rv)
    img_arr = np.expand_dims(img_arr, 2)  # (256,256)->(256,256,1)
    mask_shape = img_arr.shape
    mask1 = np.zeros(mask_shape)
    mask2 = np.zeros(mask_shape)
    mask3 = np.zeros(mask_shape)
    mask4 = np.zeros(mask_shape)
    mask1[img_arr == myo] = 1  # MYO
    mask2[img_arr == lv] = 1  # LV
    mask3[img_arr == rv] = 1  # RV
    mask4[img_arr < bg_max] = 1  # BackGround
    mask = np.concatenate([mask1, mask2, mask3, mask4], 2)  # (len,height,class_num = 4)
    return mask

def get_data():
    '''
    把图像划分训练集 验证集，要求工作路径下存在train、valid文件夹
    :return: train_list, valid_list
    '''
    slice_resize = (240,240)
    train_list = list()
    valid_list = list()

    train_png_list = os.listdir('Dataset/Dataset/train/png')
    print('-----------------------------------------')
    print('getting data: train')
    for train_png in tqdm(train_png_list):
        train_mask_path = f'Dataset/Dataset/train/mask/{train_png}'
        mask = Image.open(train_mask_path)
        mask_resize = mask.resize(slice_resize, 0)
        train_list.append(mask_resize)  # 样本一，resize
        transform1 = transforms.CenterCrop(slice_resize)
        train_list.append(transform1(mask))  # 样本二，中心剪裁
        random_rotate = randint(1, 360)
        train_list.append( mask_resize.rotate(random_rotate))  # 样本三，随机旋转
        train_list.append(mask_resize.transpose(Image.FLIP_LEFT_RIGHT))  # 样本四，左右翻转
        train_list.append(mask_resize.transpose(Image.FLIP_TOP_BOTTOM))  # 样本五，上下翻转

    valid_png_list = os.listdir('Dataset/Dataset/valid/png')
    print('-----------------------------------------')
    print('getting data: valid')
    for valid_png in tqdm(valid_png_list):
        valid_mask_path = f'Dataset/Dataset/valid/mask/{valid_png}'
        mask = Image.open(valid_mask_path)
        valid_list.append(mask.resize(slice_resize, 0))

    return train_list, valid_list

class MyDataset(Dataset):  #
    '''
    继承了torch.utils.data.Dataset,用于加载数据，后续载入神经网络中
    '''

    def __init__(self, data,TensorTransform):
        self.data = data
        self.TensorTransform = TensorTransform


    def __getitem__(self, item):  # 这个是Dataset类的关键函数，形成数据的最终形式，通过迭代的形式喂给后续的神经网络
        mask = self.data[item]
        mask = mask_one_hot(np.asarray(mask))

        return self.TensorTransform(mask) # 因为是encoding-decoding，所以只返回本身一个mask-array，既作为x，也作为y

    def __len__(self):
        return len(self.data)

class nn_processor:
    def __init__(self,train_loader,valid_loader = None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
    def train(self,net,lr=0.01,EPOCH=40,max_iter=500,save_iter=100,first_iter=0,loss_func = nn.BCEWithLogitsLoss()):
        optimizer = torch.optim.Adam(net.parameters(), lr=lr) # 在optimizer这里指明训练auto-encoder网络的参数
        i = 0
        stop = False
        for epoch in range(EPOCH):
            if stop == True:
                break
            for step, mask in enumerate(self.train_loader):
                # x, y = x.to(device), y.to(device)
                mask = mask.to(device)
                net = net.to(device)
                mask = mask.to(torch.float)
                output = net(mask)
                # output = output.to(torch.float)
                loss = loss_func(mask,output) # 编码-解码
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1

                if i % 100 == 0:
                    print(f'epoch:{epoch+1}\niteration: {i+first_iter}')
                    if i == max_iter:  # 达到最大迭代，保存模型
                        stop = True
                        torch.save(net.state_dict(), f'/kaggle/working/{i+first_iter}.pth')
                        print('\nmodel saved!')
                        break
                    if i % save_iter == 0:  # 临时保存
                        if i != save_iter:
                            os.remove(f'/kaggle/working/{i+first_iter-save_iter}.pth')
                        torch.save(net.state_dict(), f'/kaggle/working/{i+first_iter}.pth')
                        print(f'\nmodel temp {i+first_iter} saved!')
                    for data in self.valid_loader:
                        data = data.to(device)
                        data = data.to(torch.float)
                        output2 = net(data)
                        valid_loss = loss_func(output2, data)
                        print('\ntrain_loss:', float(loss))
                        print('\n-----valid_loss-----:', float(valid_loss))
                        break

if __name__ == '__main__':
    class_num = 4 # 1个后景+3个前景
    batch_size = 16

    train_list, valid_list = get_data()  # 对train_list的图像做图像增强
    TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
        transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
    ])
    train_data = MyDataset(train_list, TensorTransform=TensorTransform)
    valid_data = MyDataset(valid_list, TensorTransform=TensorTransform)  # 从image2tentor
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                              num_workers=0)  # batch_size是从这里的DataLoader传递进去的
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, num_workers=0)
    net = En_Decoder(class_num)
    try:
        model_name = get_oldest_model_name()

        net.load_state_dict(torch.load(model_name))
        print(f'\nload {model_name} successfully')
        first_iter = int(model_name.replace('.pth', '').replace('encoder/', ''))
    except Exception as er:
        print(er)
        print('load weight fail! train from no weight...')
        first_iter = 0
    encoder_processor = nn_processor(train_loader, valid_loader)
    encoder_processor.train(net,lr=0.01, EPOCH=400, max_iter=200000, first_iter=first_iter)
