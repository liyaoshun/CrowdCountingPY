# -*- coding: utf-8 -*-
import os
import time
import cv2
#import math
import random
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import Json_read as jr
#from PIL import Image, ImageEnhance
#import scipy.io as scio
#random.seed(0)
from skimage import draw#,io

DATA_DIM = 300
train_datas = []
test_datas = []
#img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
#img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

class Settings(object):
    def __init__(self, mean_value,img_dot_path2,img_box_path2,img_dot_path1,img_box_path1,img_dot_den_path="None",img_box_den_path="None"):
        '''
        设置参数
        :param mean_value:vgg训练图像的均值
               img_dot_path:dot图像路径
               img_box_path：box图像路径
               img_dot_den_path：dot密度估计文件路径
               img_box_den_path：box密度估计文件路径
        '''
        self._img_mean = np.array(mean_value)[np.newaxis, np.newaxis,:].astype(
            'float32')
        self._img_dot_path2 =img_dot_path2
        self._img_box_path2 =img_box_path2
        
        self._img_dot_path1 =img_dot_path1
        self._img_box_path1 =img_box_path1
        
        self._img_dot_den_path =img_dot_den_path
        self._img_box_den_path =img_box_den_path
    @property
    def img_mean(self):
        return self._img_mean
    @property
    def img_dot_path2(self):
        return self._img_dot_path2
    @property
    def img_box_path2(self):
        return self._img_box_path2
    @property
    def img_dot_path1(self):
        return self._img_dot_path1
    @property
    def img_box_path1(self):
        return self._img_box_path1
    @property
    def img_dot_den_path(self):
        return self._img_dot_den_path
    @property
    def img_box_den_path(self):
        return self._img_box_den_path
class Pandas_fm(object):
    def __init__(self,json_path1,json_path2):
#        json_path1 = "/media/gzs/T/gzs/baidu_star_2018_test_stage1/baidu_star_2018/annotation/annotation_train_stage1.json"
#        json_path = "/media/gzs/T/gzs/baidu_star_2018_test_stage1/baidu_star_2018/annotation/annotation_train_stage2.json"
        annotations = jr.read_all(json_path2)
        annotations1 = jr.read_all(json_path1)
        name_arr = []
        ignore_arr = []
        typ_arr = []
        id_arr = []
        stage_arr = []
        for anon in annotations:
            name = anon['name'].strip().split('/')[-1]
            name_arr.append(name)
            ignore_region = anon['ignore_region']
            ignore_arr.append(ignore_region)
            ttype = anon['type']
            typ_arr.append(ttype)
            iid = anon['id']
            id_arr.append(iid)
            stage_arr.append("stage2")
        for anon in annotations1:
            ttype = anon['type']#if the type is dot .will be contain in train setting
            if ttype == 'dot':
                name = anon['name'].strip().split('/')[-1]
                name_arr.append(name)
                ignore_region = anon['ignore_region']
                ignore_arr.append(ignore_region)
                typ_arr.append(ttype)
                iid = anon['id']
                id_arr.append(iid)
                stage_arr.append("stage1")
            else:
                pass
        pd_name = pd.Series(name_arr)
        pd_ignore = pd.Series(ignore_arr)
        pd_type = pd.Series(typ_arr)
        pd_id = pd.Series(id_arr)
        stage = pd.Series(stage_arr)
        self.pd_DataFrame = pd.DataFrame({"name":pd_name,"ignore":pd_ignore,"type":pd_type,"id":pd_id,"stage":stage})
        self.test_datas = []
        self.train_datas = []

    def split_train_test(self):
        print("split_train_test    split_train_test    split_train_test   split_train_test")
        length = self.pd_DataFrame.shape[0]
        indices = range(length)
        random.seed(time.time())
        random.shuffle(indices)

        length = len(indices)
        len_10_percent = int(length*0.1)

        self.test_datas = indices[0:len_10_percent]
        self.train_datas = indices[len_10_percent:]


def mask(image,den,ig_nore):
    Y=[]
    X=[]
    for p in ig_nore[0]:
        Y.append(p['y'])
        X.append(p['x'])
    den = np.array([den,den,den]).transpose(1,2,0)

    rr, cc=draw.polygon(np.array(Y),np.array(X))
    draw.set_color(image,[rr,cc],[0,0,0])
    draw.set_color(den,[rr,cc],[0,0,0])
    return image,den[:,:,0]

def reshape_img(image,den,w,h):

    if w > 1000 and h > 1000:
        n_w = int(w/16)*16
        n_h = int(h/16)*16
        n_w_2 = int(n_w/2)
        n_h_2 = int(n_h/2)
        if n_w_2>1000 or n_h_2>1000:
            n_w_2 = int(n_w_2/2)
            n_h_2 = int(n_h_2/2)
        image = cv2.resize(image,(n_w_2,n_h_2),interpolation=cv2.INTER_AREA)
        n_w_8 = float(n_w_2/8)
        n_h_8 = float(n_h_2/8)

        den = cv2.resize(den,(int(n_w_8),int(n_h_8)),interpolation=cv2.INTER_AREA)
#                print(den.shape)
        den = den*float((w*h)/(n_w_8*n_h_8))
    else:
        if w > h:
            n_w = float(640.0)
            n_h = float(480.0)
            image = cv2.resize(image,(int(n_w),int(n_h)),interpolation=cv2.INTER_AREA)
#                print(image.shape)
            n_w_8 = float(n_w/8)
            n_h_8 = float(n_h/8)

            den = cv2.resize(den,(int(n_w_8),int(n_h_8)),interpolation=cv2.INTER_AREA)
#                print(den.shape)
            den = den*float((w*h)/(n_w_8*n_h_8))
        elif h > w:
            n_h = float(640.0)
            n_w = float(480.0)
            image = cv2.resize(image,(int(n_w),int(n_h)),interpolation=cv2.INTER_AREA)
#                print(image.shape)
            n_w_8 = float(n_w/8)
            n_h_8 = float(n_h/8)

            den = cv2.resize(den,(int(n_w_8),int(n_h_8)),interpolation=cv2.INTER_AREA)
#                print(den.shape)
            den = den*float((w*h)/(n_w_8*n_h_8))
        else:
            n_h = float(640.0)
            n_w = float(640.0)
            image = cv2.resize(image,(int(n_w),int(n_h)),interpolation=cv2.INTER_AREA)
#                print(image.shape)
            n_w_8 = float(n_w/8)
            n_h_8 = float(n_h/8)

            den = cv2.resize(den,(int(n_w_8),int(n_h_8)),interpolation=cv2.INTER_AREA)
#                print(den.shape)
            den = den*float((w*h)/(n_w_8*n_h_8))

    return image,den

#没有使用这个函数，保留
def read_test_for_csv(data_args,panda_cl):
    pd_DataFrame = panda_cl.pd_DataFrame
    test_datas = panda_cl.test_datas
    dot_root = data_args.img_dot_path2#"/media/gzs/T/gzs/baidu_star_2018_test_stage1/baidu_star_2018/image/stage2/fixed_data/datas/dot/"
    box_root = data_args.img_box_path2#"/media/gzs/T/gzs/baidu_star_2018_test_stage1/baidu_star_2018/image/stage2/fixed_data/datas/box/"
    dot_root1 = data_args.img_dot_path1#"/media/gzs/T/gzs/baidu_star_2018_test_stage1/baidu_star_2018/image/stage1/fixed_data/datas/dot/"
    box_root1 = data_args.img_box_path1
    idl = [54,517,2625,1070,1983,2852,2237,222,211,2015]#clear error label
    for index in test_datas:
        ttype = pd_DataFrame.iloc[index]['type']
        name = pd_DataFrame.iloc[index]['name']
        ig_nore = pd_DataFrame.iloc[index]['ignore']
        iid = pd_DataFrame.iloc[index]['id']
        stage = pd_DataFrame.iloc[index]['stage']
        if stage == "stage2":
            if iid in idl:
                continue
            else:
                pass
        else:
            pass
#            print(type(ig_nore))
        len_ig = len(ig_nore)
        if ttype == 'dot' and stage == "stage2":
            img_path = dot_root + "train/" + name
            den_path = dot_root + "train_den/" + name.split('.')[0]+'.csv'
        elif  ttype == 'bbox' and stage == "stage2":
            img_path = box_root + "train/" + name
            den_path = box_root + "train_den/" + name.split('.')[0]+'.csv'
        elif  ttype == 'dot' and stage == "stage1":
            img_path = dot_root1 + "train/" + name
            den_path = dot_root1 + "train_den/" + name.split('.')[0]+'.csv'
        elif  ttype == 'bbox' and stage == "stage1":
            img_path = box_root1 + "train/" + name
            den_path = box_root1 + "train_den/" + name.split('.')[0]+'.csv'
        else:
            print(ttype)
            print(stage)
        if not os.path.exists(img_path):
#            print(img_path)
            continue
        else:
            pass
        if not os.path.exists(den_path):
            continue
        else:
            pass
        image = cv2.imread(img_path,1)
        image = image.astype(np.float32, copy=False)

        image -= data_args.img_mean
#        image /= 255.0

        den = pd.read_csv(den_path, sep=',',header=None).values
        den = den.astype(np.float32, copy=False)

        if len_ig > 0:
            image,den = mask(image,den,ig_nore)
        else:
            pass
#            print(img_path)
#            exit()
        w = float(image.shape[1])
        h = float(image.shape[0])

        image,den = reshape_img(image,den,w,h)


        image = image.astype('float32')

        image = image.transpose(2,0,1)


        image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
#        den = den.reshape((1,den.shape[0],den.shape[1]))

        den = den.astype('float32')

        yield image.astype('float32'), den.astype('float32')
def read_train_for_csv(data_args,panda_cl):
    dot_root = data_args.img_dot_path2#"/media/gzs/T/gzs/baidu_star_2018_test_stage1/baidu_star_2018/image/stage2/fixed_data/datas/dot/"
    box_root = data_args.img_box_path2#"/media/gzs/T/gzs/baidu_star_2018_test_stage1/baidu_star_2018/image/stage2/fixed_data/datas/box/"
    dot_root1 = data_args.img_dot_path1#"/media/gzs/T/gzs/baidu_star_2018_test_stage1/baidu_star_2018/image/stage1/fixed_data/datas/dot/"
    box_root1 = data_args.img_box_path1#"/media/gzs/T/gzs/baidu_star_2018_test_stage1/baidu_star_2018/image/stage1/fixed_data/datas/box/"
    idl = [54,517,2625,1070,1983,2852,2237,222,211,2015]
    def reader():
#        panda_cl = Pandas_fm()
        panda_cl.split_train_test()

        pd_DataFrame = panda_cl.pd_DataFrame

        train_datas = panda_cl.train_datas
#        print(len(train_datas))
#        exit()
        for index in train_datas:
            ttype = pd_DataFrame.iloc[index]['type']
            name = pd_DataFrame.iloc[index]['name']
            ig_nore = pd_DataFrame.iloc[index]['ignore']
            iid = pd_DataFrame.iloc[index]['id']
            stage = pd_DataFrame.iloc[index]['stage']
            if stage == "stage2":
                if iid in idl:
                    continue
                else:
                    pass
            else:
                pass
#            print(type(ig_nore))
            len_ig = len(ig_nore)
            if ttype == 'dot' and stage == "stage2":
                img_path = dot_root + "train/" + name
                den_path = dot_root + "train_den/" + name.split('.')[0]+'.csv'
            elif  ttype == 'bbox' and stage == "stage2":
                img_path = box_root + "train/" + name
                den_path = box_root + "train_den/" + name.split('.')[0]+'.csv'
            elif  ttype == 'dot' and stage == "stage1":
                img_path = dot_root1 + "train/" + name
                den_path = dot_root1 + "train_den/" + name.split('.')[0]+'.csv'
            elif  ttype == 'bbox' and stage == "stage1":
                img_path = box_root1 + "train/" + name
                den_path = box_root1 + "train_den/" + name.split('.')[0]+'.csv'
            else:
                print(ttype)
                print(stage)
            if not os.path.exists(img_path):
                print("not exists path : "+img_path)
                continue
            else:
                pass
            if not os.path.exists(den_path):
                continue
            else:
                pass

            image = cv2.imread(img_path,1)
            image = image.astype(np.float32, copy=False)

            image -= data_args.img_mean
#            image /= 255.0

            den = pd.read_csv(den_path, sep=',',header=None).values
            den = den.astype(np.float32, copy=False)

            if len_ig > 0:
                image,den = mask(image,den,ig_nore)
            else:
                pass
#            print(img_path)
#            exit()
            w = float(image.shape[1])
            h = float(image.shape[0])

            image,den = reshape_img(image,den,w,h)

            image = image.astype('float32')

            image = image.transpose(2,0,1)

            image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))

            den = den.astype('float32')
#            den = den.reshape((den.shape[0],den.shape[1]))
            yield image.astype('float32'), den.astype('float32')

    return reader


if __name__=="__main__":
    pass
