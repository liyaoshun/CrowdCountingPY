# -*- coding: utf-8 -*-
"""
Created on %(date)s
训练网络
@author: %(lys)s
"""
from __future__ import print_function
import paddle.fluid as fluid
import os
import numpy as np
import paddle
import datetime
import fluid_vgg_net as fvnet
import data_provider1 as provider
import sys

def if_exist(var):
    return os.path.exists(os.path.join(pretrained_model, var.name))

def main(data_args,use_cuda,num_passes,lr,json_path1,json_path2):
    i_shape=[3,224, 224]
    image = fluid.layers.data(name="image", shape=i_shape, dtype='float32')
    Net = fvnet.vgg_fluid()
    out = Net.net(input=image)

    label = fluid.layers.data(name="label", shape=[28*28], dtype='float32')#28*28 2

    sec = fluid.layers.square_error_cost(input=out, label=label)

    avg_cost = fluid.layers.mean(x=sec)

    optimizer = fluid.optimizer.Adam(learning_rate=lr)#Adam
    optimizer.minimize(avg_cost)

    fluid.memory_optimize(fluid.default_main_program())


    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if pretrained_model:
        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)
    panda_cl = provider.Pandas_fm(json_path1,json_path2)
    train_reader = paddle.batch(provider.read_train_for(data_args,panda_cl), batch_size=1)

    Loss = []
    ttime = 0

    for pass_id in range(num_passes):
        Loss_A = []
        for batch_id , data in enumerate(train_reader()):
            start = datetime.datetime.now()

            i_datas = np.array(data[0][0])
            i_labels = np.array(data[0][1])

            loss  = exe.run(fluid.default_main_program(),
                                  feed={"image":i_datas,"label":i_labels},fetch_list = [avg_cost])

            loss = np.mean(np.array(loss))

            Loss_A.append(loss)
            end = datetime.datetime.now()
            ttime += (end-start).total_seconds()
            if batch_id % 100 == 0:
                print("Pass {0}, trainbatch {1}, loss {2}, time {3}".format(pass_id, \
                       batch_id, loss,ttime))
                sys.stdout.flush()
                ttime = 0
        Loss.append(np.mean(Loss_A))
        np.save('./models/loss/'+str(pass_id)+'_loss.npy',np.array(Loss))
        model_path = os.path.join("./models/",str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_inference_model(model_path, ['image'], [out], exe)
    np.save('./models/loss.npy',np.array(Loss))

def start():
    #img_dot_path2  使用matlab生成的dot文件夹；dot下面的train和train_den文件夹(stage2训练数据)
    #img_box_path2  使用matlab生成的dot文件夹；box下面的train和train_den文件夹(stage2训练数据)
    #img_dot_path1  使用matlab生成的dot文件夹；dot下面的train和train_den文件夹(stage1训练数据)
    #img_box_path1  使用matlab生成的dot文件夹；box下面的train和train_den文件夹(stage1训练数据)
    setting = provider.Settings(mean_value=[104, 117, 123],
                                img_dot_path2="/media/gzs/T/gzs/baidu_star_2018_test_stage1/baidu_star_2018/image/stage2/fixed_data/datas/dot/",
                                img_box_path2="/media/gzs/T/gzs/baidu_star_2018_test_stage1/baidu_star_2018/image/stage2/fixed_data/datas/box/",
                                img_dot_path1="/media/gzs/T/gzs/baidu_star_2018_test_stage1/baidu_star_2018/image/stage1/fixed_data/datas/dot/",
                                img_box_path2="/media/gzs/T/gzs/baidu_star_2018_test_stage1/baidu_star_2018/image/stage1/fixed_data/datas/box/")
    #json_path1 是第一阶段训练集的标记文件
    #json_path2 是第二阶段训练集的标记文件

    json_path1 = "/media/gzs/T/gzs/baidu_star_2018_test_stage1/baidu_star_2018/annotation/annotation_train_stage1.json"
    json_path2 = "/media/gzs/T/gzs/baidu_star_2018_test_stage1/baidu_star_2018/annotation/annotation_train_stage2.json"
    main(data_args=setting,use_cuda=True,num_passes=170,lr=1e-6,json_path1=json_path1,json_path2=json_path2)

pretrained_model = './vgg_ilsvrc_16_fc_reduced/'
'''
run steps:
1.准备数据
在ImagePreprocessing.py中配置路径分别得到stage1和stage2数据的标记数据。（将bbox类型数据转为dot类型数据）。
然后将得到的数据分别放入stage1matlab和stage2matlab下。
配置运行stage1matlab文件夹下的gtcreat、gtcreatebox文件，得到gt数据。
配置运行stage2matlab文件夹下的gtcreat、gtcreatebox文件，得到gt数据。

配置dilated_train_fluid.py文件运行训练模型
'''


if __name__=="__main__":
    start()