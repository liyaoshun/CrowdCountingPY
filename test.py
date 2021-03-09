# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(lys)s
"""
#import sys
#sys.path.append('../scripts/')
#sys.path.append('../utils/')
import Json_read as jr
import paddle.fluid as fluid
#import paddle.v2 as paddle
#from PIL import Image
import numpy as np
#import pandas as pd
import cv2
#import os
import csv
import matplotlib.pyplot as plt
import math
#import data_provider as dp
#import data_provider1 as provider
from skimage import draw

def get_crop_box_image(image):
    w = image.shape[1]
    h = image.shape[0]
    half_w = int(w/2)
    half_h = int(h/2)
#    qure_w = int(w/4)
#    qure_h = int(h/4)
    left_up = image[0:half_h,0:half_w]
    right_up = image[0:half_h,half_w:w]
    left_down = image[half_h:h,0:half_w]
    right_down = image[half_h:h,half_w:w]
#    center = image[qure_h:half_h+qure_h,qure_w:half_w+qure_w,:]
    return left_up,right_up,left_down,right_down#,center
def get_test_path():
    path = '/media/gzs/baidu_star_2018_test_stage2/baidu_star_2018/annotation/annotation_test_stage2.json'
    annotations=jr.read_all(path)
#    print(len(annotations))
#    exit()
    return annotations
#取最接近的兩個數據做均值
def get_nest(a,b,c):
    a_b = float(abs(a-b))
    a_c = float(abs(a-c))
    b_c = float(abs(b-c))

    if (a_b > a_c ) and (a_c >b_c):
        return (b+c)/2.0
    elif (a_b > b_c) and b_c >a_c:
        return (a+c)/2.0
    elif (a_c > b_c) and b_c > a_b:
        return (a+b)/2.0
    elif a_c>a_b and a_b > b_c:
        return (b+c)/2.0
    elif b_c > a_b and a_b > a_c:
        return (a+c)/2.0
    elif b_c>a_c and a_c > a_b:
        return (a+b)/2.0
    else:
        pass

    return (a+b+c)/3.0
def reshape_img(image,w,h):

    if w > 1000 and h > 1000:
        n_w = int(w/16)*16
        n_h = int(h/16)*16
        n_w_2 = int(n_w/2)
        n_h_2 = int(n_h/2)
        if n_w_2>1000 or n_h_2>1000:
            n_w_2 = int(n_w_2/2)
            n_h_2 = int(n_h_2/2)
        image = cv2.resize(image,(n_w_2,n_h_2),interpolation=cv2.INTER_AREA)

    else:
        if w > h:
            n_w = float(640.0)
            n_h = float(480.0)
            image = cv2.resize(image,(int(n_w),int(n_h)),interpolation=cv2.INTER_AREA)

        elif h > w:
            n_h = float(640.0)
            n_w = float(480.0)
            image = cv2.resize(image,(int(n_w),int(n_h)),interpolation=cv2.INTER_AREA)

        else:
            n_h = float(640.0)
            n_w = float(640.0)
            image = cv2.resize(image,(int(n_w),int(n_h)),interpolation=cv2.INTER_AREA)


    return image
def read_image(path,ig_nore):
    image = cv2.imread(path,1)
    image = image.astype(np.float32, copy=False)
    mask(image,ig_nore)
    image -= mean

    w = float(image.shape[1])
    h = float(image.shape[0])

    image,den = reshape_img(image,w,h)
    image = image.transpose(2,0,1)

    image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))

    den = den.astype('float32')
    return image,den
def mask(image,ig_nore):
    Y=[]
    X=[]
    for p in ig_nore[0]:
        Y.append(p['y'])
        X.append(p['x'])
    rr, cc=draw.polygon(np.array(Y),np.array(X))
    draw.set_color(image,[rr,cc],[0,0,0])
    return image
def infer(root,use_cuda, model_path):
    # 是否使用GPU
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    # 生成调试器
    exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # 加载模型
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_path, exe)
        # 获取预测数据
        annotations = get_test_path()

        csvfile = open("baidu_stage2.csv","w")#将结果保存在当前文件夹下
        writer = csv.writer(csvfile)
        writer.writerow(["id","predicted"])

        for annot in annotations:
            name = annot['name']
            path = root+name.split('/')[2]
            aid = annot['id']
            ig_nore = annot['ignore_region']
            image,_ = read_image(path,ig_nore)
            i_datas = image
            results = exe.run(inference_program,
                          feed={feed_target_names[0]: i_datas},
                          fetch_list=fetch_targets)
            sm = sum(sum(np.array(results)[0,0,0,:,:]))

            writer.writerow([int(aid),int(math.ceil(sm))])
        csvfile.close()
def rotate(image, angle, center=None, scale=1.0): #1
    (h, w) = image.shape[:2] #2
    if center is None: #3
        center = (w // 2, h // 2) #4

    M = cv2.getRotationMatrix2D(center, angle, scale) #5

    rotated = cv2.warpAffine(image, M, (w, h)) #6
    return rotated #7
def show_loss():
    path = "./models/loss/52_loss.npy"
    data = np.load(path)
    plt.figure(1)
    plt.plot(data,'red')
    plt.show()
    print(data.shape)
if __name__ == '__main__':

    mean = np.array([104, 117, 123])[np.newaxis, np.newaxis,:].astype('float32')
    index = 38
    model_path = "./models/"+str(index)+"/"
    #测试图片的根路径
    root = '/media/gzs/baidu_star_2018_test_stage2/baidu_star_2018/image/stage2/test/'
    infer(root,True, model_path)
#    show_loss()

