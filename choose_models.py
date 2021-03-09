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
import pandas as pd
import cv2
import os
#import csv
import matplotlib.pyplot as plt
#import math
#import data_provider as dp
import data_provider1 as provider
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
    path = '/media/gzs/baidu_star_2018/annotation/annotation_train_stage2.json'
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
def read_image(path,ig_nore,path_den):
#    path = "/media/gzs/baidu_star_2018/image/stage2/fixed_data/datas/dot/train/0c82fb033df6b0fee1ed24a1c3723d39.jpg"
    image = cv2.imread(path,1)
    image = image.astype(np.float32, copy=False)
    if len(ig_nore)>0:
        mask(image,ig_nore)
    image -= mean
#    image /= 255.0

#    path_den = path_den#root+ttype+"/train_den/"#"/media/gzs/baidu_star_2018/image/stage2/fixed_data/datas/dot/train_den/0c82fb033df6b0fee1ed24a1c3723d39.csv"
    den = pd.read_csv(path_den, sep=',',header=None).values
    den = den.astype(np.float32, copy=False)

    w = float(image.shape[1])
    h = float(image.shape[0])

    image,den = provider.reshape_img(image,den,w,h)
#    print(image.shape)


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
def infer(root,use_cuda, model_path,index):
#    root = '/media/gzs/baidu_star_2018_test_stage1/baidu_star_2018/image/stage1/test/'
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

#        csvfile = open("baidu_stage2.csv","w")
#        writer = csv.writer(csvfile)
#        writer.writerow(["id","predicted"])
        N = 0
        E_al = 0
        for annot in annotations:
            ttype = annot['type']
            name = annot['name']
            aid = annot['id']
            num = annot['num']
            if ttype == "bbox":
                path = root+"box"+"/train/"+name.split('/')[2]
                path_den = root+"box"+"/train_den/"+name.split('/')[2].split('.')[0]+".csv"
            else:
                path = root+"dot"+"/train/"+name.split('/')[2]
                path_den = root+"dot"+"/train_den/"+name.split('/')[2].split('.')[0]+".csv"
            if os.path.exists(path):
                pass
            else:
                continue
            ig_nore = annot['ignore_region']
            image,den = read_image(path,ig_nore,path_den)
            i_datas = image#image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
            results = exe.run(inference_program,
                          feed={feed_target_names[0]: i_datas},
                          fetch_list=fetch_targets)
            sm = sum(sum(np.array(results)[0,0,0,:,:]))
            sden = sum(sum(den))
            print("index  : "+str(index)+"   num:  "+str(num)+"   id : "+str(aid)+"  pre : "+str(sm)+"  sden :"+str(sden))
            E = abs(round(sm)-round(sden))/round(sden)
            E_al = E_al + E
            N=N+1
#            plt.figure()
#            plt.subplot(121)
#            b,g,r = cv2.split(cv2.imread(path,1))
#            plt.imshow(cv2.merge([r,g,b]))
#            plt.subplot(122)
#
#            plt.imshow(np.array(results)[0,0,0,:,:])
#            plt.title("id   "+str(aid)+"   predict : "+str(round(sm)))
#            plt.show()
#            exit()
#            writer.writerow([int(aid),int(round(sm))])
        mer = float(E_al)/float(N)
        print("index  : "+str(index)+"  Merror : "+str(mer))
#        csvfile.close()
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
    root = '/media/gzs/baidu_star_2018/image/stage2/fixed_data/datas/'
    start = 15
    end = 50
    for i in range(start,end):
        model_path = "./models/"+str(i)+"/"
        infer(root,True, model_path,i)
#    show_loss()

