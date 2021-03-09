实验复现步骤：

1.准备数据
在ImagePreprocessing.py中配置路径分别得到stage1和stage2数据的标记数据。（将bbox类型数据转为dot类型数据）。并将数据分别保存在dot和box目录下
然后将得到的数据分别放入stage1matlab和stage2matlab下。
配置运行stage1matlab文件夹下的gtcreat、gtcreatebox文件，得到gt数据。
配置运行stage2matlab文件夹下的gtcreat、gtcreatebox文件，得到gt数据。
下载vgg16预训练权重(http://paddlemodels.bj.bcebos.com/vgg_ilsvrc_16_fc_reduced.tar.gz)到vgg_ilsvrc_16_fc_reduced文件夹下（并解压）
2.训练模型
配置dilated_train_fluid.py文件运行训练模型
3.选择模型测试（我们在epoch==38的时候有最好的效果）
choose_models.py
4.微调(在使用epoch==38初始权重的情况下训练)
修改lr=1e-7微调模型（分别训练小于30和大于60的数据），在epoch60次左右有最好的效果。
5.模型选择并测试
最后epoch==38的基础上给测试机一个粗估计值，然后在微调上面得到最总结果.
test.py
