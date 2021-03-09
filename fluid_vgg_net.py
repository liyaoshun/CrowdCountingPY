# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(lys)s
"""
from __future__ import print_function
import paddle.fluid as fluid

'''
构建网络结构
'''
class vgg_fluid(object):
    def __init__(self):
#        self.params = train_parameters
        self.layers = 16
    def net(self, input):
        layers = self.layers
        vgg_spec = {
            11: ([1, 1, 2, 2, 2]),
            13: ([2, 2, 2, 2, 2]),
            16: ([2, 2, 3, 3, 3]),
            19: ([2, 2, 4, 4, 4])
        }
        assert layers in vgg_spec.keys(), \
            "supported layers are {} but input layer is {}".format(vgg_spec.keys(), layers)

        nums = vgg_spec[layers]
        conv1 = self.conv_block(input, 64, nums[0])
        conv2 = self.conv_block(conv1, 128, nums[1])
        conv3 = self.conv_block(conv2, 256, nums[2])
        self.conv4 = self.conv_block_no_pool(conv3, 512, nums[3])

        conv5_1 =self.conv_bn_layer(self.conv4,ch_out=512,filter_size=3,stride=1,padding=2,dilation=2,c_str="conv5_1")
        conv5_2 = self.conv_bn_layer(conv5_1,ch_out=512,filter_size=3,stride=1,padding=2,dilation=2,c_str="conv5_2")
        conv5_3 = self.conv_bn_layer(conv5_2,ch_out=512,filter_size=3,stride=1,padding=2,dilation=2,c_str="conv5_3")
        conv6 = self.conv_bn_layer(conv5_3,ch_out=256,filter_size=3,stride=1,padding=2,dilation=2,c_str="conv6")
        conv7 = self.conv_bn_layer(conv6,ch_out=128,filter_size=3,stride=1,padding=2,dilation=2,c_str="conv7")
        conv8 = self.conv_bn_layer(conv7,ch_out=64,filter_size=3,stride=1,padding=2,dilation=2,c_str="conv8")
        out = self.conv_bn_layer(conv8,ch_out=1,filter_size=1,stride=1,padding=0,dilation=1,c_str="out")
        return out
    def conv_bn_layer(self,input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  c_str,
                  dilation=1,
                  act='relu',
                  bias_attr=False):
        tmp = fluid.layers.conv2d(
            input=input,
            name = c_str,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            dilation = dilation,
            param_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Normal(scale=0.01)),
            bias_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Constant(value=0.0)))
        nor = fluid.layers.batch_norm(input=tmp)
        return  fluid.layers.relu(nor)

    def conv_block(self, input, num_filter, groups):
        conv = input
        for i in range(groups):
            conv = fluid.layers.conv2d(
                input=conv,
                num_filters=num_filter,
                filter_size=3,
                stride=1,
                padding=1,
                act='relu',
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Normal(scale=0.01),trainable=False),
                bias_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Constant(value=0.0),trainable=False))
        return fluid.layers.pool2d(
            input=conv, pool_size=2, pool_type='max', pool_stride=2)
    def conv_block_no_pool(self, input, num_filter, groups):
        conv = input
        for i in range(groups):
            conv = fluid.layers.conv2d(
                input=conv,
                num_filters=num_filter,
                filter_size=3,
                stride=1,
                padding=1,
                act='relu',
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Normal(scale=0.01),trainable=False),
                bias_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Constant(value=0.0),trainable=False))
        return conv