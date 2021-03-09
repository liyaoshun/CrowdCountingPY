# -*- coding: utf-8 -*-
import numpy as np
np.set_printoptions(threshold='nan')
import Json_read as jr
import cv2
#import math
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

import matplotlib.patches as patches
from skimage import draw,io
import os
import scipy.io as scio
import pandas as pd
#import json
"""
Created on %(date)s
处理百度训练图形  将图像box变为dot类型
@author: %(lys)s
root 所有原始训练数据存放位置
anon_path 原始标注文件所在位置
box_path  box类型数据放置的位置
dot_path dot类型数据放置的位置
"""

class ImagePreprocess(object):
    def __init__(self,root,anon_path,box_path,dot_path):
        self.anon_path = anon_path
        self.root = root
        self.box_path = box_path
        self.dot_path = dot_path
    '''
    显示原始标记数据
    将dot 和 box类型的数据使用点和方框在图像上显示出来
    '''
    def show_point(self):
        idl = [54,517,2625,1070,1983,2852,2237,222,211,2015]
        root = '/media/gzs/baidu_star_2018/image/stage2/'
        path = self.anon_path#'/media/gzs/baidu_star_2018/annotation/annotation_train_stage1.json'
        data = jr.read(path)#keys  :   points   name   type
        length = len(data)
        plt.ioff()
        for i in range(0,length):
            iid = data[i]['id']

            if iid in idl:
                continue
            else:
                pass
#            print(i)
            points = np.array(data[i]['points'])
            length = points.shape[0]
            type_t = data[i]['type']
            name = data[i]['name']#.split('.')[0]+'.jpg'
            is_show = False

            for nnn in cls.read23():
                if str(name) == str(nnn[0]):
                    is_show = True
                    break
                else:
                    pass

            if is_show == True:
                pass
            else:
                continue

            if type_t == 'dot':
                img_path = root +"dot/"+ name
            else:
                img_path = root +"box/"+ name
            image = cv2.imread(img_path,0)
            plt.close('all')
            plt.figure(1)
            plt.imshow(image,cmap=plt.cm.gray)
            if type_t == 'dot':
                plt.plot(points[:,0],points[:,1],'ro')

            elif type_t == 'bbox':
                print('ignore : '+str(len(data[i]['ignore_region'])))#ignore_region
                plt.plot(points[:,0],points[:,1],'ro')
                currentAxis=plt.gca()
                for point in points:
                    rect=patches.Rectangle((point[0], point[1]),point[2],point[3],linewidth=1,edgecolor='r',facecolor='none')
                    currentAxis.add_patch(rect)
            plt.title("type : "+str(type_t)+"  id :"+str(data[i]['id'])+" len : "+str(len(points)))
            plt.show()

    '''
    将数据分为dot和box两类数据分别保存
    '''
    def split_data(self):
        path = self.anon_path
        root = self.root
        save_path_dot = self.dot_path
        save_path_box = self.box_path
        Data = jr.read_all(path)
        index = 0
        if not os.path.exists(save_path_box):
            os.mkdir(save_path_box)
        else:
            pass
        if not os.path.exists(save_path_dot):
            os.mkdir(save_path_dot)
        else:
            pass
        for d in Data:
            img_path =root+d['name'].split('/')[2]
            print(d['name'])
            img = io.imread(img_path)
            s_path=''
            if d['type'] == 'dot':
                s_path = save_path_dot + d['name'].split('/')[2]
            elif d['type']=='bbox':
                s_path = save_path_box + d['name'].split('/')[2]
            else:
                print("********************error*******************")
            io.imsave(s_path,img)
            index += 1
    '''
    移除数据集中标记忽略区域
    将标记的忽略区域设置为（255,255,255）
    '''
    def removemask(self):
        path = self.anon_path#'/media/gzs/baidu_star_2018/annotation/annotation_train_stage1.json'
        root = self.root  #'/media/gzs/baidu_star_2018/image/'
        save_path_dot = self.dot_path #'/media/gzs/baidu_star_2018/image/stage1/dot/'
        save_path_box = self.box_path #'/media/gzs/baidu_star_2018/image/stage1/box/'
        #    path = '/media/gzs/baidu_star_2018_test_stage1/baidu_star_2018/annotation/annotation_test_stage1.json'
        Data = jr.read_all(path)
        index = 0
        if not os.path.exists(save_path_box):
            os.mkdir(save_path_box)
        else:
            pass
        if not os.path.exists(save_path_dot):
            os.mkdir(save_path_dot)
        else:
            pass
        for d in Data:
            if len(d['ignore_region'])>0:
                img_path =root+d['name'].split('/')[2]
                img = io.imread(img_path)
                ignore = d['ignore_region']
                print(d['name'])
                Y=[]
                X=[]
                for p in ignore[0]:
                    Y.append(p['y'])
                    X.append(p['x'])

                rr, cc=draw.polygon(np.array(Y),np.array(X))
                draw.set_color(img,[rr,cc],[255,255,255])#[255,255,255]
                s_path=''
                if d['type'] == 'dot':
                    s_path = save_path_dot + d['name'].split('/')[2]
                elif d['type']=='bbox':
                    print("***********************bbox***********************")
                    s_path = save_path_box + d['name'].split('/')[2]
                else:
                    print("********************error*******************")
                io.imsave(s_path,img)
            else:
                img_path =root+d['name'].split('/')[2]
                print(d['name'])
                img = io.imread(img_path)
                s_path=''
                if d['type'] == 'dot':
                    s_path = save_path_dot + d['name'].split('/')[2]
                elif d['type']=='bbox':
                    s_path = save_path_box + d['name'].split('/')[2]
                else:
                    print("********************error*******************")
                io.imsave(s_path,img)
            index += 1
#            if index > 10:
#                exit()
#            else:
#                pass
    '''
    生成人群密度估计图
    使用的是matlab代码
    '''
    def generate_density(self):
        pass
    '''
    w_o  h_o  是原图的宽和高
    去掉单个图像中的我们方法很难识别的图像
    '''
    def Cleaning_points(self,w_o,h_o,points):
        leave_rst = []
        mask_rst = []
        for p in points:
            w = p[2]
            h = p[3]
            div_1_9 = float(1)/float(9)
#            print('h: '+str(h)+"   w : "+str(w))
            if w == 0:
                continue
            if h == 0:
                continue
            div_h_w = float(h)/float(w)
            div_4_3 = float(4)/float(3)

            if w < div_1_9*w_o:
                leave_rst.append(p)
            elif div_h_w>div_4_3:
                leave_rst.append(p)
            elif h < div_1_9*h_o:
                mask_rst.append(p)
            else:
                mask_rst.append(p)
    #    print(mask_rst)
        return leave_rst,mask_rst
    def Direct_points(self,w_o,h_o,points):
        leave_rst = []
        mask_rst = []
        for p in points:

            leave_rst.append(p)
    #    print(mask_rst)
        return leave_rst,mask_rst
    '''
    将box类型数据中一些很难自动将box框转换为dot的去掉
    及:框出来的box的h/w  小于4:3 的 且框大于当前图片宽度的1/9
    root表示当前box类型数据存放的父路径
    anon_path 表示标记文件所在的路径
    '''
    def CleaningData(self):#root='/media/gzs/baidu_star_2018/image/stage1/box/'
        path = self.anon_path#anon_path
        data = jr.read(path)
        rst = []

        for d in data:
            type_t = d['type']
            if type_t == 'dot':
                continue
            else:
                name = d['name']
                is_show = False
                for nnn in cls.read23():
                    if str(name) == str(nnn[0]):
                        is_show = True
                        break
                    else:
                        pass
                if is_show == True:
                    pass
                else:
                    continue
                path = self.box_path + d['name']
                image = cv2.imread(path,0)

                h = image.shape[0]
                w = image.shape[1]
#                print("o_w : "+str(w)+"  o_h : "+str(h))
                points_t = d['points']
#                print(len(points_t))
#                print(points_t)
                points,mask_points = self.Cleaning_points(w,h,points_t)

                if len(points) == 0:
                    tcl = {'name':d['name'],'points':points_t,'mask':mask_points,'id':d['id']}
                else:
                    tcl = {'name':d['name'],'points':points,'mask':mask_points,'id':d['id']}
                rst.append(tcl)
        return rst
    '''
    将box类型数据转换为dot类型数据
    为了使用box和dot两类数据，需要将box数据转换为dot类型
    '''
    def trans_box_to_dot(self):
        Data = self.CleaningData() #将box类型数据中一些很难自动将box框转换为dot的去掉
        rst = []
        for d in Data:
            points = d['points']
            tp = []
            for p in points:
                w = p[2]
                w_21 = w/2
                h = p[3]
                h_41 = h/7
                x = p[0] + w_21
                y = p[1] + h_41
                tp.append([x,y])
            d['points'] = tp
            d['mask'] = points
            rst.append(d)
        return rst
    '''
    将原始标记中的box类型标记读取出来
    '''
    def get_box_Data(self):
        path = self.anon_path  #'/media/gzs/baidu_star_2018/annotation/annotation_train_stage1.json'
        f = open(path, "rb")
        fileJson = jr.json.load(f)#fileJson  keys  : [u'info', u'stage', u'annotations', u'split']
        annotations = fileJson['annotations']
        Data = []
        for ano in annotations:
            if ano['type'] == 'bbox':
                tp = {'name':ano['name'],'num':ano['num']}
                Data.append(tp)
            else:
                pass
        return Data

    def get_anon_dot(self):
        path = self.anon_path  #'/media/gzs/baidu_star_2018/annotation/annotation_train_stage1.json'
        Data = jr.read(path)
        rst = []
        for d in Data:
            if d['type'] == 'dot':
                rst.append(d)
            else:
                pass
        return rst
    def show_point_box(self):
        root = '/media/gzs/baidu_star_2018/image/stage2/train/'
#        data = jr.read(path)#keys  :   points   name   type
        data =  np.load("clean.npy")#cls.trans_box_to_dot()#scio.loadmat("/media/gzs/baidu_star_2018/image/stage2/box.mat")

        length = len(data)
        for i in range(0,length):
            points = np.array(data[i]['points'])
#            if len(points)<150:
#                continue
#            else:
#                pass
            type_t = data[i]['type']
            name = data[i]['name']#.split('.')[0]+'.jpg'
            img_path = root + name
#            print(img_path)
            image = cv2.imread(img_path,0)
            plt.figure(0)
            plt.imshow(image,cmap=plt.cm.gray)
            if type_t == 'dot':
                plt.plot(points[:,0],points[:,1],'ro')
            elif type_t == 'bbox':
#                continue
                print('ignore : '+str(len(data[i]['ignore_region'])))#ignore_region
                plt.plot(points[:,0],points[:,1],'ro')
                currentAxis=plt.gca()
                for point in points:
                    rect=patches.Rectangle((point[0], point[1]),point[2],point[3],linewidth=1,edgecolor='r',facecolor='none')
                    currentAxis.add_patch(rect)
            plt.title("  id :"+str(data[i]['id'])+" len : "+str(len(points)))
            plt.show()

    #当图像小与300之后，需要将图像边框补齐300+
    def resize_image_adjust(self,image=None, height=None, width=None):
#         image =cv2.imread("/media/gzs/baidu_star_2018/image/stage2/train/d4cc72c34d161a224b57257a9afa0930.jpg",1)
         top, bottom, left, right = (0, 0, 0, 0)
         #获取图像尺寸
         h, w, _ = image.shape
         #对于长宽不相等的图片，找到最长的一边
         longest_edge = max(h, w)
         #计算短边需要增加多上像素宽度使其与长边等长
         if h < longest_edge:
             dh = longest_edge - h
             top = dh // 2
             bottom = dh - top
         elif w < longest_edge:
             dw = longest_edge - w
             left = dw // 2
             right = dw - left
         else:
             pass
         #RGB颜色
         BLACK = [0, 0, 0]
         #给图像增加边界，使图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
         constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
         if height == None or width == None:
             return constant
         else:
             return cv2.resize(constant, (height, width))
    def show_img_den_val(self):
        img_path = "/media/gzs/baidu_star_2018/image/stage2/fixed_data/datas/dot/train/"
        den_path = "/media/gzs/baidu_star_2018/image/stage2/fixed_data/datas/dot/train_den/"
        img_list = os.listdir(img_path)
        for imgl in img_list:
            img = cv2.imread(img_path+imgl,0)
            den = pd.read_csv(den_path+imgl.split('.')[0]+".csv",sep=',',header=None).values
            plt.figure(0)
            plt.subplot(121)
            plt.imshow(img,cmap=plt.cm.gray)
            plt.subplot(122)
            plt.imshow(den)
            plt.title("name : "+imgl)
            plt.show()
    def get_expansion_shape(self,w,h):
        if w>=h:
            if w > 960:
                n_w = 960
                scale = float(n_w) / float(w)
                n_h = (int(h * scale)/8)*8
                return n_w,n_h
            else :
                n_w = int(float(w)/8.0)*8
                n_h = int(float(h)/8.0)*8
                return n_w,n_h
        else:
            if h > 960:
                n_h = 960
                scale = float(n_h) / float(h)
                n_w = (int(w * scale)/8)*8
                return n_w,n_h
            else:
                n_w = int(float(w)/8.0)*8
                n_h = int(float(h)/8.0)*8
                return n_w,n_h

    def argument_data(self):
        _anon_path = self.anon_path
#        _root = self.root
        _box_img_path = "/media/gzs/baidu_star_2018/image/stage2/fixed_data/datas/box/train/"
        _box_den_path = "/media/gzs/baidu_star_2018/image/stage2/fixed_data/datas/box/train_den/"

        _dot_img_path = "/media/gzs/baidu_star_2018/image/stage1/fixed_datas/datas/dot/train/"
        _dot_den_path = "/media/gzs/baidu_star_2018/image/stage1/fixed_datas/datas/dot/train_den/"

        save_image_path = "/media/gzs/baidu_star_2018/image/stage2/fixed_data/datas/image/"
        save_den_path = "/media/gzs/baidu_star_2018/image/stage2/fixed_data/datas/den/"

        data = jr.read(_anon_path)
        for d in data:
            name = d['name']
            dtype = d['type']
            points = d['points']
            if 'dot' == dtype:
                img_path = _dot_img_path + name
                den_path = _dot_den_path + name.split('.')[0]+".csv"
            elif 'bbox' == dtype:
                continue
#                img_path = _box_img_path + name
#                den_path = _box_den_path + name.split('.')[0]+".csv"
            image = cv2.imread(img_path)
            density = pd.read_csv(den_path,sep=',',header=None).values
#            print(image.shape)
#            print(density.shape)
#            print(density)
#            print(density[np.where(density>0)])
#            exit()
            plt.figure(0)
            plt.subplot(121)
            plt.imshow(image,cmap=plt.cm.gray)
            plt.subplot(122)
            plt.imshow(density)
            plt.title("name : "+str(sum(sum(density))))
            plt.show()
#            exit()
#            w = image.shape[1]
#            h = image.shape[0]
#            _w,_h = self.get_expansion_shape(w,h)
#
#            img_new = cv2.resize(image,(_w,_h))
#            _w_8 = int(_w/8)
#            _h_8 = int(_h/8)
#            sv_img = save_image_path + name
#            sv_den = save_den_path + name.split('.')[0]+".npy"
#            print(d['num'])
#            print(sum(sum(density)))
#            density = cv2.resize(density,(_w_8,_h_8))
#            density = density*float((float(w)*float(h))/(float(_h_8)*float(_w_8)))
#            print(sum(sum(density)))
#            print(sv_img)
#            print(sv_den)
#            exit()
#            cv2.imwrite(sv_img,img_new)
#            np.save(sv_den,density)

    def show_wh(self):
        _anon_path = self.anon_path
#        _root = self.root
        _box_path = self.box_path
        _dot_path = self.dot_path
        data = jr.read(_anon_path)
        for d in data:
            name = d['name']
            dtype = d['type']
#            points = d['points']
            if 'dot' == dtype:
                img_path = _dot_path + name
            else:
                img_path = _box_path + name
            image = cv2.imread(img_path)
            w = image.shape[1]
            h = image.shape[0]

            print("w : "+str(w)+"  h : "+str(h))
    def read23(self):
        path = "/media/gzs/baidu_star_2018/image/stage2/fixed_data/scripts/23.mat"
        mat_data = scio.loadmat(path)
        return mat_data['namedata'][0]
    def show_23(self):
        root = "/media/gzs/baidu_star_2018/image/stage2/fixed_data/datas/box/train_den_23/"
        dt_list = os.listdir(root)
        for nm in dt_list:
            path = root + nm
            img = pd.read_csv(path,sep=',',header=None).values
            plt.figure()
            plt.imshow(img)
            plt.show()
    def show_test(self):
        path = '/media/gzs/baidu_star_2018_test_stage2/baidu_star_2018/annotation/annotation_test_stage2.json'
        Data = jr.read_all(path)
        for d in Data:
            print(d['ignore_region'])
if __name__=="__main__":
    '''
    参数定义请看ImagePreprocess参数解析
    '''
    dot_path = '/media/gzs/baidu_star_2018/image/stage2/dot/'
    box_path = '/media/gzs/baidu_star_2018/image/stage2/box/'
    anon_path = '/media/gzs/baidu_star_2018/annotation/annotation_train_stage2.json'

    root = '/media/gzs/baidu_star_2018/image/stage2/train/'

    cls = ImagePreprocess(root,anon_path,box_path,dot_path)

    '''
    step one
    #移除mask数据
    #并将数据分为box 和 dot两类数据
    cls.removemask()
    '''

    '''
    #step two
    #转换box数据为dot数据
    rst =  cls.trans_box_to_dot()#将清理后的数据保存下来
    保存box类型标注为mat格式数据以便在matlab中使用
    scio.savemat("box.mat", {'data':rst})
    '''
#    rst =  cls.trans_box_to_dot()
#    scio.savemat("/media/gzs/baidu_star_2018/image/stage2/23_box.mat", {'data':rst})
    '''
    #step three
    保存dot类型标注为mat格式数据以便在matlab中使用
    cls.get_anon_dot()
    '''
#    rst = cls.get_anon_dot()
#    scio.savemat("/media/gzs/baidu_star_2018/image/stage2/dot.mat", {'points':np.array(rst)})