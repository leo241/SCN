import nibabel as nib  # 处理.nii类型图片
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from random import randint

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from unet import *
from auto_encoder import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")# 使用gpu加速

import warnings

warnings.filterwarnings("ignore")  # ignore warnings


class data_processor:
    def __init__(self):
        self.myo = 125
        self.lv = 220
        self.rv = 255
        self.bg_max = min(self.myo, self.lv, self.rv)
        self.slice_resize = (240, 240)

    def nii_fig_id2dir(self, id):
        '''根据患者id 找到原nii路径'''
        return f'origin data/lge/patient{id}_LGE.nii.gz'

    def nii_mask_id2dir(self, id):
        '''根据患者id 找到原nii mask路径'''
        return f'origin data/mask/patient{id}_LGE_manual.nii.gz'

    def png_fig_dir(self, dataset, patient_id, slice_id):
        return f'Dataset/{dataset}/png/patient{patient_id}_{slice_id}.png'

    def png_mask_dir(self, dataset, patient_id, slice_id):
        return f'Dataset/{dataset}/mask/patient{patient_id}_{slice_id}.png'

    def imshow_dir2d(self, dir2d):
        '''给定某个2d图片路径，plt显示它的图像'''
        img = Image.open(dir2d)
        plt.imshow(img)

    def view2plot(self, fig1, fig2, name1='fig1', name2='fig2'):
        plt.subplot(121)
        plt.imshow(fig1)
        plt.title(name1)
        plt.subplot(122)
        plt.imshow(fig2)
        plt.title(name2)
        plt.show()

    def id2dir(self, patient_id, slice_id):
        normal_show = False
        for dataset in ['train', 'valid', 'test']:
            try:
                png_dir = self.png_fig_dir(dataset, patient_id, slice_id)
                Image.open(png_dir)  # 用于试探是否真的有这个地址
                mask_dir = self.png_mask_dir(dataset, patient_id, slice_id)
                return (png_dir, mask_dir)
                normal_show = True
                break
            except Exception as er:
                # print(er)
                continue
        if not normal_show:
            print('<Error Unfound: this patient slice not found! check your input!>')
            print('-----------------------------------------')

    def view_png_and_mask(self, patient_id, slice_id):
        normal_show = False
        for dataset in ['train', 'valid', 'test']:
            try:
                png_dir = self.png_fig_dir(dataset, patient_id, slice_id)
                mask_dir = self.png_mask_dir(dataset, patient_id, slice_id)
                plt.subplot(121)
                self.imshow_dir2d(png_dir)
                plt.title(f'<{dataset}> patient{patient_id}_{slice_id} slice')
                plt.subplot(122)
                self.imshow_dir2d(mask_dir)
                plt.title(f'<{dataset}> patient{patient_id}_{slice_id} mask')
                plt.show()
                normal_show = True
                break
            except Exception as er:
                # print(er)
                continue
        if not normal_show:
            print('<Error Unfound: this patient slice not found! check your input!>')
            print('-----------------------------------------')

    def view_mask_and_arr(self, patient_id, slice_id):
        normal_show = False
        for dataset in ['train', 'valid', 'test']:
            try:
                mask_dir = self.png_mask_dir(dataset, patient_id, slice_id)
                self.imshow_dir2d(mask_dir)
                plt.title(f'<{dataset}> patient{patient_id}_{slice_id} mask')
                plt.show()
                normal_show = True
                return np.asarray(Image.open(mask_dir))
            except Exception as er:
                # print(er)
                continue
        if not normal_show:
            print('<Error Unfound: this patient slice not found! check your input!>')
            print('-----------------------------------------')

    def create_slice_dataset(self, dataset_size=[25, 5, 15]):
        '''根据已有的LGE NII数据拆分为slice的二维png数据，并进行resize
        划分训练集、验证集、测试集，储存在同目录文件<Dataset>下'''
        print('-----------------------------------------')
        print('preparing for dataset...'
              '\nplease ensure that you have folder named <origin data> in your work dir!')
        if len(os.listdir('Dataset/train/png')) | len(os.listdir('Dataset/train/mask')) | len(
                os.listdir('Dataset/valid/png')) | len(os.listdir('Dataset/valid/mask')) | len(
                os.listdir('Dataset/test/png')) | len(os.listdir('Dataset/test/mask')):
            print(
                'create dataset fail! it seems that you already have some data in <Dataset> please check!\n<Error: data already exist!>')
            print('-----------------------------------------')
            return False  # 确保Dataset文件夹下没有任何数据
        patients = np.arange(45) + 1  # 45个患者的编号列表
        train_nums = np.random.choice(patients, dataset_size[0], replace=False)  # 随机抽取25个作为训练集
        valid_nums = np.random.choice(list(set(patients) - set(train_nums)), dataset_size[1], replace=False)  # 5个作为验证集
        test_nums = list(set(patients) - set(train_nums) - set(valid_nums))  # 15个作为测试集

        for train_id in tqdm(train_nums):
            png = nib.load(self.nii_fig_id2dir(train_id)).get_data()
            mask = nib.load(self.nii_mask_id2dir(train_id)).get_data()
            mask[mask == 200] = 125
            mask[mask == 500] = 220
            mask[mask == 600] = 255
            slices = png.shape[2]  # 切片的数量
            for slice in range(slices):
                Image.fromarray(mask[:, :, slice]).convert('L').save(
                    self.png_mask_dir('train', train_id, slice))
                Image.fromarray(png[:, :, slice]).convert('L').save(
                    self.png_fig_dir('train', train_id, slice))

        for valid_id in tqdm(valid_nums):
            png = nib.load(self.nii_fig_id2dir(valid_id)).get_data()
            mask = nib.load(self.nii_mask_id2dir(valid_id)).get_data()
            mask[mask == 200] = 125
            mask[mask == 500] = 220
            mask[mask == 600] = 255
            slices = png.shape[2]  # 切片的数量
            for slice in range(slices):
                Image.fromarray(mask[:, :, slice]).convert('L').save(
                    self.png_mask_dir('valid', valid_id, slice))
                Image.fromarray(png[:, :, slice]).convert('L').save(
                    self.png_fig_dir('valid', valid_id, slice))

        for test_id in tqdm(test_nums):
            png = nib.load(self.nii_fig_id2dir(test_id)).get_data()
            mask = nib.load(self.nii_mask_id2dir(test_id)).get_data()
            mask[mask == 200] = self.myo  # MYO
            mask[mask == 500] = self.lv  # LV
            mask[mask == 600] = self.rv  # RV
            slices = png.shape[2]  # 切片的数量
            for slice in range(slices):
                Image.fromarray(mask[:, :, slice]).convert('L').save(
                    self.png_mask_dir('test', test_id, slice))
                Image.fromarray(png[:, :, slice]).convert('L').save(
                    self.png_fig_dir('test', test_id, slice))

        print('dataset created successfully!')
        print('-----------------------------------------')

    def mask_one_hot(self, img_arr):
        img_arr = np.expand_dims(img_arr, 2)  # (256,256)->(256,256,1)
        mask_shape = img_arr.shape
        mask1 = np.zeros(mask_shape)
        mask2 = np.zeros(mask_shape)
        mask3 = np.zeros(mask_shape)
        mask4 = np.zeros(mask_shape)
        mask1[img_arr == self.myo] = 1  # MYO
        mask2[img_arr == self.lv] = 1  # LV
        mask3[img_arr == self.rv] = 1  # RV
        mask4[img_arr < self.bg_max] = 1  # BackGround
        mask = np.concatenate([mask1, mask2, mask3, mask4], 2)  # (len,height,class_num = 4)
        return mask

    def get_data(self):
        '''
        把图像划分训练集 验证集，要求工作路径下存在train、valid文件夹
        :return: train_list, valid_list
        '''
        train_list = list()
        valid_list = list()

        train_png_list = os.listdir('Dataset/Dataset/train/png')
        print('-----------------------------------------')
        print('getting data: train')
        for train_png in tqdm(train_png_list):
            train_png_path = f'Dataset/Dataset/train/png/{train_png}'
            train_mask_path = f'Dataset/Dataset/train/mask/{train_png}'
            slice_num = int(train_png.replace('.png', '').split('_')[1])  # 切片的序号作为y2
            img = Image.open(train_png_path)
            # img_arr = np.expand_dims(img_arr, 2)
            mask = Image.open(train_mask_path)
            # transform0 = transforms.Resize(self.slice_resize)
            img_resize = img.resize(self.slice_resize, 0)
            mask_resize = mask.resize(self.slice_resize, 0)
            train_list.append([img_resize, mask_resize, slice_num])  # 样本一，resize
            transform1 = transforms.CenterCrop(self.slice_resize)
            train_list.append([transform1(img), transform1(mask), slice_num])  # 样本二，中心剪裁
            # transform4 = transforms.Compose([transforms.,
            #                                  transforms.Resize(self.slice_resize)])
            # transform5 = transforms.Compose([transforms.RandomHorizontalFlip(),
            #                                  transforms.Resize(self.slice_resize)])
            # transform6 = transforms.Compose([transforms.RandomVerticalFlip(),
            #                                  transforms.Resize(self.slice_resize)])
            random_rotate = randint(1, 360)
            train_list.append(
                [img_resize.rotate(random_rotate), mask_resize.rotate(random_rotate), slice_num])  # 样本三，随机旋转
            train_list.append(
                [img_resize.transpose(Image.FLIP_LEFT_RIGHT), mask_resize.transpose(Image.FLIP_LEFT_RIGHT),
                 slice_num])  # 样本四，左右翻转
            train_list.append(
                [img_resize.transpose(Image.FLIP_TOP_BOTTOM), mask_resize.transpose(Image.FLIP_TOP_BOTTOM),
                 slice_num])  # 样本五，上下翻转

        valid_png_list = os.listdir('Dataset/Dataset/valid/png')
        print('-----------------------------------------')
        print('getting data: valid')
        for valid_png in tqdm(valid_png_list):
            valid_png_path = f'Dataset/Dataset/valid/png/{valid_png}'
            valid_mask_path = f'Dataset/Dataset/valid/mask/{valid_png}'
            slice_num = int(valid_png.replace('.png', '').split('_')[1])  # 切片的序号作为y2
            img = Image.open(valid_png_path)
            # img_arr = np.expand_dims(img_arr, 2)
            mask = Image.open(valid_mask_path)
            valid_list.append([img.resize(self.slice_resize, 0), mask.resize(self.slice_resize, 0), slice_num])

        return train_list, valid_list


    def dice_score(self, fig1, fig2, class_value):
        '''
        计算某种特定像素级类别的DICE SCORE
        :param fig1:
        :param fig2:
        :param class_value:
        :return:
        '''
        fig1_class = fig1 == class_value
        fig2_class = fig2 == class_value
        A = np.sum(fig1_class)
        B = np.sum(fig2_class)
        AB = np.sum(fig1_class & fig2_class)
        if A + B == 0:
            return 1
        return 2 * AB / (A + B)

    def dice_score_between_real_and_predict(self, real, predict):
        myo_score = self.dice_score(real, predict, self.myo)
        lv_score = self.dice_score(real, predict, self.lv)
        rv_score = self.dice_score(real, predict, self.rv)
        return myo_score, lv_score, rv_score

    def dataset_mean_dice_score(self, net, dataset='test'):
        '''给定模型和数据集，预测这个数据集里面的平均dice score，包括MYO LV RV'''
        dataset_png_dir = f'Dataset/{dataset}/png'
        myo = 0
        lv = 0
        rv = 0
        all_dir = os.listdir(dataset_png_dir)
        list_myo = list()
        list_lv = list()
        list_rv = list()
        # length = len(all_dir)
        print(f'counting {dataset} dice score, please waiting...')
        for png_dir in tqdm(all_dir):
            real_mask_dir = f'Dataset/Dataset/{dataset}/mask/{png_dir}'
            real_mask = np.asarray(Image.open(real_mask_dir))
            predict_mask = self.predict(net, f'Dataset/Dataset/{dataset}/png/{png_dir}', self.slice_resize)
            myo_score, lv_score, rv_score = self.dice_score_between_real_and_predict(real_mask, predict_mask)
            if np.sum(real_mask == self.myo) == 0 or np.sum(real_mask == self.lv) == 0 or np.sum(
                    real_mask == self.rv) == 0:
                continue
            list_myo.append(myo_score)
            list_lv.append(lv_score)
            list_rv.append(rv_score)
            # if np.sum(real_mask==self.myo) != 0:
            #     list_myo.append(myo_score)
            # if np.sum(real_mask==self.lv) != 0:
            #     list_lv.append(lv_score)
            # if np.sum(real_mask==self.rv) != 0:
            #     list_rv.append(rv_score)
        return list_myo, list_lv, list_rv

        #     if np.sum(real_mask==self.myo) == 0 or np.sum(real_mask==self.lv) == 0 or np.sum(real_mask==self.rv) == 0:
        #         length -= 1
        #         continue
        #     predict_mask = self.predict(net, f'Dataset/{dataset}/png/{png_dir}',self.slice_resize)
        #     myo_score, lv_score, rv_score = self.dice_score_between_real_and_predict(real_mask,predict_mask)
        #     myo += myo_score
        #     lv += lv_score
        #     rv += rv_score
        # print(f'{dataset} length',length)
        # return myo/length, lv/length, rv/length
        #

    def predict(self, net, target, slice_resize):
        '''
        给定模型和图片，以及网络预测所需要的resize，预测mask，返回mask矩阵
        :param net:
        :param target:
        :return:
        '''
        if type(target) == str:
            img_target = Image.open(target)
            origin_size = img_target.size
            img_arr = np.asarray(img_target.resize(slice_resize, 0))
        elif type(target) == PngImageFile:
            origin_size = target.size
            img_arr = np.asarray(target.resize(slice_resize, 0))
        elif type(target) == np.ndarray:
            origin_size = target.shape
            img_arr = np.asarray(Image.fromarray(target).resize(slice_resize, 0))
        else:
            print('<target type error>')
            return False
        TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
            transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
        ])
        img_tensor = TensorTransform(img_arr)
        img_tensor4d = img_tensor.unsqueeze(0)  # 只有把图像3维（1，256，256）扩展成4维（1，1，256，256）才能放进神经网络预测

        print(target, net(img_tensor4d)[1])
        predict = net(img_tensor4d)[0].squeeze(0)
        predict_tag = torch.max(predict, 0).indices.data.numpy()

        predict_mask = np.zeros(predict_tag.shape)

        predict_mask[predict_tag == 0] = self.myo  # myo
        predict_mask[predict_tag == 1] = self.lv  # lv
        predict_mask[predict_tag == 2] = self.rv  # rv
        predict_mask = np.asarray(Image.fromarray(predict_mask).resize(origin_size, 0))
        return predict_mask


class MyDataset(Dataset):  #
    '''
    继承了torch.utils.data.Dataset,用于加载数据，后续载入神经网络中
    '''

    def __init__(self, data, TensorTransform):
        self.data = data
        self.TensorTransform = TensorTransform

    def __getitem__(self, item):  # 这个是Dataset类的关键函数，形成数据的最终形式，通过迭代的形式喂给后续的神经网络
        img, mask, slice_num = self.data[item]
        img_arr = np.asarray(img)
        img_arr = np.expand_dims(img_arr, 2)  # (256,256)->(256,256,1) # 实际图像矩阵
        mask = data_processor().mask_one_hot(np.asarray(mask))

        return self.TensorTransform(img_arr), self.TensorTransform(mask), torch.tensor(slice_num)

    def __len__(self):
        return len(self.data)


class nn_processor:
    def __init__(self, train_loader, valid_loader=None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def train(self, net, lr=0.01, EPOCH=40, max_iter=500, save_iter=500, plot_iter=2000, first_iter=0,
              loss_func=nn.BCEWithLogitsLoss(), loss_func2=nn.MSELoss()):
        net = net.to(device)
        o_model = 'bencoder.pth'  # 预训练得到的自编码-解码模型
        encoder = En_Decoder(4)
        encoder.load_state_dict(torch.load(o_model))  # 加载预训练好的自编码网络
        encoder = encoder.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 这里没有加入encoder.parameters(),所以自编码网络的参数不会改变
        i = 0
        loss_train_list = list()
        loss_valid_list = list()
        iter_list = list()
        stop = False
        for epoch in range(EPOCH):
            if stop == True:
                break
            for step, (x, y, y2) in enumerate(self.train_loader):
                x, y,y2 = x.to(device), y.to(device),y2.to(device)
                output1, output2 = net(x)
                output1 = output1.to(torch.float)
                y = y.to(torch.float)
                output2 = output2.to(torch.float)
                y2 = y2.to(torch.float)
                loss = loss_func(output1, y) * 30 + loss_func2(output2, y2).to(torch.float) * 0.01 + loss_func2(
                    encoder(output1), encoder(y))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1

                if i % 100 == 0:
                    print(f'epoch:{epoch+1}\niteration: {i+first_iter}')
                    if i == max_iter:  # 达到最大迭代，保存模型
                        stop = True
                        torch.save(net.state_dict(), f'/kaggle/working/{i+first_iter}.pth')
                        print('model saved!')
                        break
                    if i % save_iter == 0:  # 临时保存
                        if i!= save_iter:
                            os.remove(f'/kaggle/working/{i+first_iter-save_iter}.pth')
                        torch.save(net.state_dict(), f'/kaggle/working/{i+first_iter}.pth')
                        print(f'model temp {i+first_iter} saved!')
                    for data in self.valid_loader:
                        x_valid, y_valid, slice_valid = data
                        x_valid = x_valid.to(device)
                        y_valid = y_valid.to(device)
                        slice_valid = slice_valid.to(device)
                        output1, output2 = net(x_valid)
                        valid_loss = loss_func(output1, y_valid)
                        loss_train_list.append(float(loss))  # 每隔10个iter，记录一下当前train loss
                        loss_valid_list.append(float(valid_loss))  # 每隔10个iter，记录一下当前valid loss
                        iter_list.append(i + first_iter)  # 记录当前的迭代次数
                        print('train_loss:', float(loss))
                        print('-----valid_loss-----:', float(valid_loss))
                        break


if __name__ == '__main__':
    Mydata_processor = data_processor()  # 创建数据处理器
    Mydata_processor.create_slice_dataset()  # 根据<origin data>文件夹划分训练集、验证集、测试集的slice pn
    # Mydata_processor.view_png_and_mask(patient_id=15,slice_id=12) # 查看某位病人的某张切片图像和其mask对比图
    # mask_arr = Mydata_processor.view_mask_and_arr(patient_id=10, slice_id=0) # 查看某位病人mask图片并且返回矩阵
    # 看一下图像增强的结果：
    # name = ['resize','resize crop','rotation','hori','vertical']
    # j = 15
    # train_list, valid_list = Mydata_processor.get_data()  # 对train_list的图像做图像增强
    # for i in np.arange(5):
    #     Mydata_processor.view2plot(train_list[i+5*j][0],train_list[i+5*j][1],name[i]+str(i+5*j),name[i]+str(i+5*j))
    batch_size = 8  # 设置部分超参数
    class_num = 4


    def train():
        train_list, valid_list = Mydata_processor.get_data()  # 对train_list的图像做图像增强
        TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
            transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
        ])

        train_data = MyDataset(train_list, TensorTransform=TensorTransform)
        valid_data = MyDataset(valid_list, TensorTransform=TensorTransform)  # 从image2tentor

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                  num_workers=0)  # batch_size是从这里的DataLoader传递进去的
        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, num_workers=0)
        net = UNet(class_num)
        try:
            model_name = Mydata_processor.get_oldest_model_name()
            net.load_state_dict(torch.load(model_name))
            first_iter = int(model_name.replace('.pth', '').replace('model_save/', ''))
        except Exception as er:
            # print(er)
            print('load weight fail! train from no weight...')
            first_iter = 0
        unet_processor = nn_processor(train_loader, valid_loader)
        unet_processor.train(net, EPOCH=400, max_iter=200000, first_iter=first_iter)


    train()


    def view_model_effect_change(patient, slice):
        net = UNet(class_num)
        png_dir, mask_dir = Mydata_processor.id2dir(patient, slice)
        model_name_list = os.listdir('model_save')
        number_list = list()
        for model_name in model_name_list:
            number_list.append(int(model_name.replace('.pth', '')))
        number_list.sort()
        for number in number_list:
            if number == 86600:
                net.load_state_dict(torch.load(f'model_save/{number}.pth'))
                predict_mask = Mydata_processor.predict(net, png_dir, Mydata_processor.slice_resize)
                real_mask = np.asarray(Image.open(mask_dir))
                myo, lv, rv = Mydata_processor.dice_score_between_real_and_predict(real_mask, predict_mask)
                Mydata_processor.view2plot(predict_mask, real_mask,
                                           f'{number}iter myo:{round(myo,2)} lv:{round(lv,2)} rv:{round(rv,2)}',
                                           f'patient{patient} slice{slice}')


    # view_model_effect_change(patient=3,slice=6)
    # test_patients = os.listdir('Dataset/test/mask')
    # for test_patient in test_patients:
    #     num_list = test_patient.replace('.png','').replace('patient','').split('_')
    #     view_model_effect_change(num_list[0],num_list[1])

    # view_model_effect_change(patient=24, slice=14)
    def dataset_dice_score():
        model_name = Mydata_processor.get_oldest_model_name()  # 获得迄今为止训练次数最多的模型
        # model_name = 'model_save/45000.pth'
        print(model_name)
        net = UNet(class_num)
        net.load_state_dict(torch.load(model_name))
        train_dice = Mydata_processor.dataset_mean_dice_score(net, 'train')
        print('train_dice:', train_dice)
        valid_dice = Mydata_processor.dataset_mean_dice_score(net, 'valid')
        print('valid_dice:', valid_dice)
        test_dice = Mydata_processor.dataset_mean_dice_score(net, 'test')
        print('test_dice:', test_dice)
        return test_dice


    # test_dice = dataset_dice_score()
    def view_valid_dice_change(model1, model2='MAX', step=1):
        if model2 == 'MAX':
            model2 = int(Mydata_processor.get_oldest_model_name().replace('model_save/', '').replace('.pth', ''))
        net = UNet(class_num)
        iter_list = list()
        myo_list = list()
        lv_list = list()
        rv_list = list()
        for model_name in tqdm(range(model1, model2 + 1, step * 100)):
            net.load_state_dict(torch.load(f'model_save/{model_name}.pth'))
            iter_list.append(model_name)
            myo, lv, rv = Mydata_processor.dataset_mean_dice_score(net, 'valid')
            myo_list.append(myo)
            lv_list.append(lv)
            rv_list.append(rv)
        plt.plot(iter_list, myo_list, label='myo')
        plt.plot(iter_list, lv_list, label='lv')
        plt.plot(iter_list, rv_list, label='rv')
        plt.xlabel('iteration')
        plt.ylabel('valid dice score')
        plt.legend()  # 显示图例文字
        plt.title(f'valid dice score with iteration')
        plt.savefig(f'valid dice score plot/vds from {model1} to {model2}.png')
        return iter_list, myo_list, lv_list, rv_list


    # view_valid_dice_change(88100,step=15)
    # iter_list, myo_list, lv_list, rv_list = view_valid_dice_change(model1=89200, step=10)
    net = UNet(class_num)
    # net.load_state_dict(torch.load(Mydata_processor.get_oldest_model_name()))
    net.load_state_dict(torch.load('model_save/99600.pth'))
    # net.load_state_dict(torch.load('model_save/80600.pth'))
    # net.load_state_dict(torch.load('model_save/159100.pth'))
    # myo,lv,rv = Mydata_processor.dataset_mean_dice_score(net,'test')
    # print(np.mean(myo),np.std(myo))
    # print(np.mean(lv),np.std(lv))
    # print(np.mean(rv),np.std(rv))



