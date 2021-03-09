# -*- coding: utf-8 -*-
"""
Created on %(date)s
读取 json 文件
@author: %(lys)s
"""
#from pandas.io.json import json_normalize
#import pandas as pd
import json
#import time
import numpy as np
#import scipy.io as scio
import matplotlib.pyplot as plt
def read(path):
    f = open(path, "rb")
    fileJson = json.load(f)#fileJson  keys  : [u'info', u'stage', u'annotations', u'split']
    annotations = fileJson['annotations']#  keys：  : name:当前标记的图片名   id:当前是第几个数据   num:共有几个区域或点   ignore_region   type(dot   bbox)   annotation
    all_length = len(annotations)
    Data = []
    Dot_Count = 0
    Bbox_Count = 0
    for i in range(0,all_length):
        img_path = annotations[i]['name'].strip().split('/')[-1]
        point_len = annotations[i]['num']#['num']
        point_list = []
        type_t = annotations[i]['type']
        ignore_region = annotations[i]['ignore_region']
        if type_t == 'bbox':
            Bbox_Count += 1
            for j in range(0,point_len):
                x = annotations[i]['annotation'][j]['x']
                y = annotations[i]['annotation'][j]['y']
                w = annotations[i]['annotation'][j]['w']
                h = annotations[i]['annotation'][j]['h']
                point_list.append([x,y,w,h])
        elif type_t == 'dot':
            Dot_Count += 1
            for j in range(0,point_len):
                x = annotations[i]['annotation'][j]['x']
                y = annotations[i]['annotation'][j]['y']
                point_list.append([x,y])
        else:
            pass
        cl = {'name':img_path,'type':type_t,'points':point_list,'ignore_region':ignore_region,'id':annotations[i]['id'],'num':point_len}
#        print(annotations[i])
        Data.append(cl)
#    print("Bbox_Count : "+str(Bbox_Count)+"   Dot_Count : "+str(Dot_Count))
    return Data

def read_all(path):
    f = open(path, "rb")
    fileJson = json.load(f)#fileJson  keys  : [u'info', u'stage', u'annotations', u'split']
    annotations = fileJson['annotations']#  keys：  : name:当前标记的图片名   id:当前是第几个数据   num:共有几个区域或点   ignore_region   type(dot   bbox)   annotation
    return annotations
def draw_hist(data,title):
    plt.figure(0)
    plt.hist(data,100)
    plt.title(title)
    plt.show()
def statistical(path):
    annotations = read_all(path)
    all_length = len(annotations)
    box_num = []
    dot_num = []
    for i in range(0,all_length):
        _type = annotations[i]["type"]
        num = annotations[i]["num"]
        if "dot" == _type:
            dot_num.append(num)
        elif "bbox" == _type:
            box_num.append(num)
        else:
            print("数据有错。请检查")
    return np.array(box_num),np.array(dot_num)
if __name__=='__main__':
    path = '/media/gzs/baidu_star_2018/annotation/annotation_train_stage2.json'
    read(path)