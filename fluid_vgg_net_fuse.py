# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(lys)s
"""
from __future__ import print_function
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
#from paddle.fluid.initializer import Xavier
from paddle.fluid.initializer import Constant
from paddle.fluid.initializer import Bilinear
from paddle.fluid.regularizer import L2Decay


def conv_bn(input, filter, ksize, stride, padding,c_str ,act='relu', bias_attr=False):
    conv = fluid.layers.conv2d(
        input=input,
        name = c_str,
        filter_size=ksize,
        num_filters=filter,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=bias_attr)
    return fluid.layers.batch_norm(input=conv, act=act)
'''
构建网络结构
'''
class vgg_fluid(object):
    def __init__(self):
        self.layers = 16
    def net_fuse(self,input):
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
        conv1 = self.conv_block_vgg(input, 64, nums[0])
        conv2 = self.conv_block_vgg(conv1, 128, nums[1])
        self.conv3 = self.conv_block_vgg_no_pool(conv2, 256, nums[2])
        pool3 = fluid.layers.pool2d(input=self.conv3, pool_size=2, pool_type='max', pool_stride=2)

        self.conv4 = self.conv_block_vgg_no_pool(pool3, 512, nums[3])

#        print(self.conv4)
        self.conv5_1 = self.conv_bn_layer(self.conv4,ch_out=1024,filter_size=3,stride=1,padding=1,dilation=1,c_str="dconv5_1")

        self.conv5_2 = self.conv_bn_layer(self.conv5_1,ch_out=512,filter_size=1,stride=1,padding=0,dilation=1,c_str="dconv5_2")

        self.conv6_1 = self.conv_bn_layer(self.conv5_2,ch_out=256,filter_size=1,stride=1,padding=0,dilation=1,c_str="dconv6_1")
        self.conv6_2 = self.conv_bn_layer(self.conv6_1,ch_out=512,filter_size=3,stride=1,padding=2,dilation=2,c_str="dconv6_2")

        self.conv7_1 = self.conv_bn_layer(self.conv6_2,ch_out=128,filter_size=1,stride=1,padding=0,dilation=1,c_str="dconv7_1")
        self.conv7_2 = self.conv_bn_layer(self.conv7_1,ch_out=256,filter_size=3,stride=1,padding=2,dilation=2,c_str="dconv7_2")

        self.pyramid_conv()

#        out = self.conv_bn_layer(self.conv7_2,ch_out=1,filter_size=1,stride=1,padding=0,dilation=1,c_str="out")
        self.conv3_out = self.conv_bn_layer(self.conv3_norm,ch_out=1,filter_size=1,stride=1,padding=0,dilation=1,c_str="conv3_out")
        self.conv3_out_pool = fluid.layers.pool2d(input=self.conv3_out, pool_size=2, pool_type='avg', pool_stride=2)
        self.conv4_out = self.conv_bn_layer(self.conv4_norm,ch_out=1,filter_size=1,stride=1,padding=0,dilation=1,c_str="conv4_out")
        self.ssh_conv5_out = self.conv_bn_layer(self.ssh_conv5,ch_out=1,filter_size=1,stride=1,padding=0,dilation=1,c_str="ssh_conv5_out")
        self.ssh_conv6_out = self.conv_bn_layer(self.ssh_conv6,ch_out=1,filter_size=1,stride=1,padding=0,dilation=1,c_str="ssh_conv6_out")
        self.ssh_conv7_out = self.conv_bn_layer(self.ssh_conv7,ch_out=1,filter_size=1,stride=1,padding=0,dilation=1,c_str="ssh_conv7_out")
        self.out_concat = fluid.layers.concat([self.conv3_out_pool,self.conv4_out,self.ssh_conv5_out,self.ssh_conv6_out,self.ssh_conv7_out],axis=1)
        self.out = self.conv_bn_layer(self.out_concat,ch_out=1,filter_size=1,stride=1,padding=0,dilation=1,c_str="out")

        return self.out #self.conv3_out_pool,self.conv4_out,self.ssh_conv5_out,self.ssh_conv6_out,self.ssh_conv7_out

    def fpn(self,up_from, up_to,name,is_sampling = False):
        ch = up_to.shape[1]
        b_attr = ParamAttr(learning_rate=2., regularizer=L2Decay(0.))
        conv1 = fluid.layers.conv2d(
            up_from, ch, 1, act='relu', bias_attr=b_attr,name = name+"_fpn_conv1")

        if is_sampling:
            w_attr = ParamAttr(
                learning_rate=0.,
                regularizer=L2Decay(0.),
                initializer=Bilinear())
            upsampling = fluid.layers.conv2d_transpose(
                conv1,
                ch,
                output_size=None,
                filter_size=4,
                padding=1,
                stride=2,
                groups=ch,
                param_attr=w_attr,
                bias_attr=False,
                use_cudnn=True,name = name+"_fpn_upsampling")
        else:
            upsampling = conv1

        b_attr = ParamAttr(learning_rate=2., regularizer=L2Decay(0.))
        conv2 = fluid.layers.conv2d(
            up_to, ch, 1, act='relu', bias_attr=b_attr,name = name+"_fpn_conv2")

        # eltwise mul
        conv_fuse = upsampling * conv2

        return fluid.layers.relu(x=conv_fuse)

    def cpm(self,input,name):

            branch1 = conv_bn(input, 1024, 1, 1, 0,name+"_br1", act=None)
            branch2a = conv_bn(input, 256, 1, 1, 0,name+"_br2a", act='relu')
            branch2b = conv_bn(branch2a, 256, 3, 1, 1,name+"_br2b", act='relu')
            branch2c = conv_bn(branch2b, 1024, 1, 1, 0,name+"_br2c", act=None)

            sum = branch1 + branch2c

            rescomb = fluid.layers.relu(x=sum)

            # ssh
            b_attr = ParamAttr(learning_rate=2., regularizer=L2Decay(0.))
            ssh_1 = fluid.layers.conv2d(rescomb, 256, 3, 1, 1, bias_attr=b_attr,name=name+"_ssh_1")
            ssh_dimred = fluid.layers.conv2d(
                rescomb, 128, 3, 1, 1, act='relu', bias_attr=b_attr,name=name+"_ssh_dimred")
            ssh_2 = fluid.layers.conv2d(
                ssh_dimred, 128, 3, 1, 1, bias_attr=b_attr,name=name+"_ssh_2")
            ssh_3a = fluid.layers.conv2d(
                ssh_dimred, 128, 3, 1, 1, act='relu', bias_attr=b_attr,name=name+"_ssh_3a")
            ssh_3b = fluid.layers.conv2d(ssh_3a, 128, 3, 1, 1, bias_attr=b_attr,name=name+"_ssh_3b")

            ssh_concat = fluid.layers.concat([ssh_1, ssh_2, ssh_3b], axis=1)

            ssh_out = fluid.layers.relu(x=ssh_concat)
            return ssh_out
    def pyramid_conv(self):

        self.lfpn1_on_conv4 = self.fpn(self.conv5_2, self.conv4,"fpn_conv4")#  -1*512*28*28

        self.lfpn0_on_conv3 = self.fpn(self.lfpn1_on_conv4, self.conv3,"fpn_conv3",is_sampling=True)#256

        self.ssh_conv3 = self.cpm(self.lfpn0_on_conv3,"cmp_conv3")
        self.ssh_conv4 = self.cpm(self.lfpn1_on_conv4,"cmp_conv4")

        self.ssh_conv5 = self.cpm(self.conv5_2,"cmp_conv5")

        self.ssh_conv6 = self.cpm(self.conv6_2,"cmp_conv6")
        self.ssh_conv7 = self.cpm(self.conv7_2,"cmp_conv7")
#        print(self.ssh_conv3)
        self.conv3_norm = self.ssh_conv3#self._l2_norm_scale(self.ssh_conv3, init_scale=10.)
        self.conv4_norm = self.ssh_conv4#self._l2_norm_scale(self.ssh_conv4, init_scale=8.)

    def _l2_norm_scale(self, input, init_scale=1.0, channel_shared=False):
        from paddle.fluid.layer_helper import LayerHelper
        helper = LayerHelper("Scale")
        l2_norm = fluid.layers.l2_normalize(
            input, axis=1)  # l2 norm along channel
        shape = [1] if channel_shared else [input.shape[1]]

        scale = helper.create_parameter(
            attr=helper.param_attr,
            shape=shape,
            dtype=input.dtype,
            default_initializer=Constant(init_scale))

        out = l2_norm*scale#fluid.layers.elementwise_mul(x=l2_norm, y=scale)
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
    def simple_conv(self,input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  c_str,
                  dilation=1,
                  act='relu',
                  bias_attr=False):
        b_attr = ParamAttr(learning_rate=2., regularizer=L2Decay(0.))
        tmp = fluid.layers.conv2d(
            input=input,
            name = c_str,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            dilation = dilation,
            bias_attr = b_attr)
        return  tmp
    def conv_block_vgg(self, input, num_filter, groups):
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
    def conv_block_vgg_no_pool(self, input, num_filter, groups):
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