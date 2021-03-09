# -*- coding: utf-8 -*-
"""
Created on %(date)s
自己转换的权重来构建的网络
@author: %(lys)s
"""
import paddle.fluid as fluid
import numpy as np
class Net(object):
    def __init__(self):
        pass
        self.layers = 16

    def inference(self,input):
        conv1_1 = self.conv_sigle(input,"conv1_1", 64)
        conv1_2 = self.conv_sigle(conv1_1,"conv1_2", 64)
        pool1 = self.pool_max(conv1_2,"pool1")
        conv2_1 = self.conv_sigle(pool1,"conv2_1", 128)
        conv2_2 = self.conv_sigle(conv2_1,"conv2_2", 128)
        pool2 = self.pool_max(conv2_2,"pool2")
        conv3_1 = self.conv_sigle(pool2,"conv3_1", 256)
        conv3_2 = self.conv_sigle(conv3_1,"conv3_2", 256)
        conv3_3 = self.conv_sigle(conv3_2,"conv3_3", 256)
        pool3 = self.pool_max(conv3_3,"pool3")
        conv4_1 = self.conv_sigle(pool3,"conv4_1", 512)
        conv4_2 = self.conv_sigle(conv4_1,"conv4_2", 512)
        conv4_3 = self.conv_sigle(conv4_2,"conv4_3", 512)

        conv5_1 =self.conv_bn_layer(conv4_3,ch_out=512,filter_size=3,stride=1,padding=2,dilation=2,c_str="dconv5_1")
        conv5_2 = self.conv_bn_layer(conv5_1,ch_out=512,filter_size=3,stride=1,padding=2,dilation=2,c_str="dconv5_2")
        conv5_3 = self.conv_bn_layer(conv5_2,ch_out=512,filter_size=3,stride=1,padding=2,dilation=2,c_str="dconv5_3")
        conv6 = self.conv_bn_layer(conv5_3,ch_out=256,filter_size=3,stride=1,padding=2,dilation=2,c_str="dconv6")
        conv7 = self.conv_bn_layer(conv6,ch_out=128,filter_size=3,stride=1,padding=2,dilation=2,c_str="dconv7")
        conv8 = self.conv_bn_layer(conv7,ch_out=64,filter_size=3,stride=1,padding=2,dilation=2,c_str="dconv8")
        out = self.conv_bn_layer(conv8,ch_out=1,filter_size=1,stride=1,padding=0,dilation=1,c_str="out")

        return out

    def conv_sigle(self , input,name, num_filter):
        conv = fluid.layers.conv2d(
            input=input,
            name = name,
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
    def pool_max(self,conv,name):
        pl = fluid.layers.pool2d(
            input=conv,name=name, pool_size=2, pool_type='max', pool_stride=2)
        return pl
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

        bn = fluid.layers.batch_norm(input=tmp)
        rel = fluid.layers.relu(bn)
#        nor = fluid.layers.batch_norm(input=tmp,epsilon=0.001)
        return  rel
    def load_weights(self,data_path,exe,place,layer_load):
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            if op_name in layer_load:
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        if param_name == "weights":
                            name = op_name+".w_0"
                        elif param_name == "biases":
                            name = op_name+".b_0"
                        v = fluid.global_scope().find_var(name)
                        w = v.get_tensor()
                        w.set(data.reshape(w.shape()), place)
                        print(str(op_name)+"  "+str(param_name)+"  加载成功")
                    except ValueError,e:
                         print(e.message)
            else:
                print("不需要加载的权重 ： "+str(op_name))
    '''
    查看caffe转过来权重的结构
    '''
    def show_npy_struct(self):
        data_path = "/media/gzs/datas/vgg16/VGG_ILSVRC_16_layers_latest.npy"
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            print(op_name)
            for param_name, data in data_dict[op_name].iteritems():
                print(param_name)
                print(type(data))
    def show_net_struct(self):
        pass
if __name__=="__main__":
    net = Net()