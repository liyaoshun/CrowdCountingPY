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
import fluid_vgg_net_fuse as fvnet
import data_provider1 as provider
import sys
#import cv2

def if_exist(var):
#    print(var.name)
    return os.path.exists(os.path.join(pretrained_model, var.name))
def calc_loss(label8,conv3_out_pool,conv4_norm,ssh_conv5,ssh_conv6,ssh_conv7):

    c3l = fluid.layers.square_error_cost(input=conv3_out_pool, label=label8)#fluid.layers.elementwise_sub(conv3_out_pool,label8,axis=1)
#    c3l.persistable = True
    c3l_mean = fluid.layers.reduce_mean(c3l)
#    c3l = c3l*c3l #fluid.layers.elementwise_mul(c3l,c3l,axis=1)
#    print(c3l_mean)
#    exit()
    c4l = fluid.layers.square_error_cost(input=conv4_norm, label=label8)#fluid.layers.elementwise_sub(conv4_norm,label8,axis=1)
#    c4l.persistable = True
#    c4l = c4l*c4l#fluid.layers.elementwise_mul(c4l,c4l,axis=1)
    c4l_mean = fluid.layers.reduce_mean(c4l)

    c5l = fluid.layers.square_error_cost(input=ssh_conv5, label=label8)#fluid.layers.elementwise_sub(ssh_conv5,label8,axis=1)
#    c5l.persistable = True
#    c5l = c5l*c5l#fluid.layers.elementwise_mul(c5l,c5l,axis=1)
    c5l_mean = fluid.layers.reduce_mean(c5l)

    c6l = fluid.layers.square_error_cost(input=ssh_conv6, label=label8)#fluid.layers.elementwise_sub(ssh_conv6,label8,axis=1)
#    c6l.persistable = True
#    c6l = c6l*c6l#fluid.layers.elementwise_mul(c6l,c6l,axis=1)
    c6l_mean = fluid.layers.reduce_mean(c6l)

    c7l = fluid.layers.square_error_cost(input=ssh_conv7, label=label8)#fluid.layers.elementwise_sub(ssh_conv7,label8,axis=1)
#    c7l.persistable = True
#    c7l = c7l*c7l#fluid.layers.elementwise_mul(c7l,c7l,axis=1)
    c7l_mean = fluid.layers.reduce_mean(c7l)

#    loss_conv3_norm = fluid.layers.reduce_sum(c3l)
#    loss_conv4_norm = fluid.layers.reduce_sum(c4l)
#    loss_ssh_conv5  = fluid.layers.reduce_sum(c5l)
#    loss_ssh_conv6  = fluid.layers.reduce_sum(c6l)
#    loss_ssh_conv7  = fluid.layers.reduce_sum(c7l)

    loss_al = c3l_mean + c4l_mean + c5l_mean + c6l_mean + c7l_mean

    loss_mean = loss_al

    return loss_mean
#only  sub mean
def main(data_args,use_cuda,num_passes,lr):
    i_shape=[3,224, 224]
    image = fluid.layers.data(name="image", shape=i_shape, dtype='float32')

    Net = fvnet.vgg_fluid()

    out = Net.net_fuse(input=image)
    out.persistable = True

    label = fluid.layers.data(name="label", shape=[2], dtype='float32')

    sec = fluid.layers.square_error_cost(input=out, label=label)

    avg_cost = fluid.layers.mean(sec)
#    loss = 
#    avg_cost = calc_loss(label,conv3_out_pool,conv4_norm,ssh_conv5,ssh_conv6,ssh_conv7)
#    batch_size = 1
#    optimizer_method = 'momentum'
#    learning_rate = 0.001
#    steps_per_pass = 4258 / batch_size
    optimizer = fluid.optimizer.Adam(learning_rate=lr)#Adam
#    boundaries = [steps_per_pass * 50, steps_per_pass * 80,
#                  steps_per_pass * 120, steps_per_pass * 140]
#    values = [
#        learning_rate, learning_rate * 0.5, learning_rate * 0.25,
#        learning_rate * 0.1, learning_rate * 0.01
#    ]
#
#    if optimizer_method == "momentum":
#        optimizer = fluid.optimizer.Momentum(
#            learning_rate=fluid.layers.piecewise_decay(
#                boundaries=boundaries, values=values),
#            momentum=0.9,
#            regularization=fluid.regularizer.L2Decay(0.0005),
#        )
#    else:
#        optimizer = fluid.optimizer.RMSProp(
#            learning_rate=fluid.layers.piecewise_decay(boundaries, values),
#            regularization=fluid.regularizer.L2Decay(0.0005),
#        )

    optimizer.minimize(avg_cost)

    fluid.memory_optimize(fluid.default_main_program())


    place = fluid.CUDAPlace(2) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
#    x = filter(if_exist, fluid.default_startup_program().list_vars())
#    print(x)
#    exit()
    if pretrained_model:
        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)
#    exit()
    panda_cl = provider.Pandas_fm()
    train_reader = paddle.batch(provider.read_train_for_csv(data_args,panda_cl), batch_size=1)

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
            if batch_id % 10 == 0:
                print("Pass {0}, trainbatch {1}, loss {2}, time {3}".format(pass_id, \
                       batch_id, loss,ttime))
                sys.stdout.flush()
                ttime = 0
        Loss.append(np.mean(Loss_A))
        np.save('./models1/loss/'+str(pass_id)+'_loss.npy',np.array(Loss))
        model_path = os.path.join("./models1/",str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        _target_vars  = [out]
        fluid.io.save_inference_model(dirname=model_path, feeded_var_names=['image'], target_vars=_target_vars, executor=exe)
    np.save('./models1/loss.npy',np.array(Loss))

def start():
    #vgg16预训练权重所在路径
    setting = provider.Settings(mean_value=[104., 117., 123.],
                                img_dot_path="/media/gzs/baidu_star_2018/image/stage2/datas/dot/train/",
                                img_box_path="/media/gzs/baidu_star_2018/image/stage2/datas/box/train/",
                                img_dot_den_path="/media/gzs/baidu_star_2018/image/stage2/datas/dot/train_den/",
                                img_box_den_path="/media/gzs/baidu_star_2018/image/stage2/datas/box/train_den/")
    main(data_args=setting,use_cuda=True,num_passes=120,lr=1e-6)

pretrained_model = '/home/gzs/Documents/denys/baidupaddleprog_stage2/train/vgg_ilsvrc_16_fc_reduced/'
if __name__=="__main__":
    start()