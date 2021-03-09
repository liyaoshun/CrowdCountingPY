%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%代码改编至上海科技大学
%box类型数据的人群密度估计图生成文件
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear ALL;
seed = 95461354;
rng(seed)
path = '/media/gzs/baidu_star_2018/image/stage2/';%使用ImagePreprocessing处理后保存的dot和box的父路径

dataset = 'box';
num_images = 1727;
% k = 3;
% N = 9;

output_path = '../datas/';%保存处理后数据的路径
train_path_img = strcat(output_path, dataset,'/train/');%处理后的图像数据路径
train_path_den = strcat(output_path, dataset,'/train_den/');%处理后的人群密度估计图路径

mkdir(output_path);
mkdir(train_path_img);
mkdir(train_path_den);

dt_data = load('./box.mat');%dot类型数据文件路径
list_points = dt_data.data;

num_val = ceil(num_images*0.1);
indices = randperm(num_images);
idl = [54,517,2625,1070,1983,2852,2237,222,211,2015];

for idx = 1:num_images
    annPoints =  list_points{1,idx}.points;
    an_len = length(annPoints);
    if an_len <=0
        continue
    end
    i = indices(idx);
    if (mod(idx,10)==0)
        fprintf(1,'Processing %3d/%d files\n', idx, num_images);
    end
    if ismember(list_points{1,idx}.id,idl) == 1
        continue
    end
    input_img_name = strcat(path,dataset,'/',list_points{1,idx}.name);
    im = imread(input_img_name);
    img = im;
    [h, w, c] = size(im);
    if (c == 3)
        im = rgb2gray(im);
    end
    wn2 = w/8; hn2 = h/8;
    wn2 =8 * floor(wn2/8);
    hn2 =8 * floor(hn2/8);


    if( w <= 2*wn2 )
        im = imresize(im,[ h,2*wn2+1]);
        annPoints(:,1) = annPoints(:,1)*2*wn2/w;
    end
    if( h <= 2*hn2)
        im = imresize(im,[2*hn2+1,w]);
        annPoints(:,2) = annPoints(:,2)*2*hn2/h;
    end

    im_density = get_density_map_gaussian(im,annPoints);%get_density_our(im,annPoints,k);
    [h, w, c] = size(im);
    a_w = wn2+1; b_w = w - wn2;
    a_h = hn2+1; b_h = h - hn2;
    name_l = strsplit(list_points{1,idx}.name,'.');

    imwrite(img, [train_path_img,list_points{1,idx}.name]);
    csvwrite([train_path_den,name_l{1,1},'.csv'], im_density);

end

