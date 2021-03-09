# -*- coding: utf-8 -*-
"""
Created on %(date)s
训练网络 将从caffe转过来的npy权重保存为paddle能直接读取的格式
@author: %(lys)s
"""
from __future__ import print_function
import paddle.fluid as fluid
import self_trains_net as fvnet
import data_provider as provider

def main(data_args,use_cuda,num_passes,lr):
    i_shape=[3,224, 224]
    image = fluid.layers.data(name="image", shape=i_shape, dtype='float32')
    Net = fvnet.Net()
    out = Net.inference(input=image)

    label = fluid.layers.data(name="label", shape=[2], dtype='float32')

    sec = fluid.layers.square_error_cost(input=out, label=label)

    avg_cost = fluid.layers.mean(x=sec)

    optimizer = fluid.optimizer.Adam(learning_rate=lr)#Adam
    optimizer.minimize(avg_cost)

    fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    layer_load = ["conv1_1","conv1_2","conv2_1","conv2_2","conv3_1","conv3_2","conv3_3","conv4_1","conv4_2","conv4_3"]
    Net.load_weights(pretrained_npy,exe,place,layer_load)

#    if pretrained_model:
#        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)
    model_path = "./pre_weight/"
    fluid.io.save_inference_model(model_path, ['image'], [out], exe)

def start():
    #vgg16预训练权重所在路径
    setting = provider.Settings(mean_value=[104, 117, 124],
                                img_dot_path="/media/gzs/baidu_star_2018/image/stage1/dot_den/dot/train/",
                                img_box_path="/media/gzs/baidu_star_2018/image/stage1/box_den/box/train/",
                                img_dot_den_path="/media/gzs/baidu_star_2018/image/stage1/dot_den/dot/train_den/",
                                img_box_den_path="/media/gzs/baidu_star_2018/image/stage1/box_den/box/train_den/")
    main(data_args=setting,use_cuda=False,num_passes=25,lr=1e-6)

pretrained_model = '/home/gzs/Documents/denys/baidupaddleprog/fluidscripts/models-develop/image_classification/caffe2paddle/vgg16_paddle/'
pretrained_npy = "/media/gzs/datas/vgg16/VGG_ILSVRC_16_layers_latest.npy"#caffe 转过来的权重
if __name__=="__main__":
    start()