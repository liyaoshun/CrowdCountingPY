function density_map = get_density_our( img, points, k )
%固定高斯核制作人群密度估计图
%

image = img;%imread(image_path);

position_head = points;
length = size(position_head,1);
%if length < 4
%    fprintf('length < 3');
%else
%    distance_mat = distance(position_head, k);
%end
if length > 150
    distance_mat = distance(position_head, k);
end
density_map = zeros(size(image,1), size(image,2));

for pid = 1:size(position_head,1)
   var = 15;%使用固定核制作人群密度估计图
   if length > 150
       var = 0.3 * mean(distance_mat(pid, :));%标准差
        if var <=0
            continue
        end
   end
   

    
    ph = [floor(position_head(pid,2)), floor(position_head(pid,1))];
    dh = norm2d(image, [var, var], [double(ph(1)),double(ph(2))]);
    dh = dh./sum(sum(dh));
    density_map = dh + density_map;
end
end

function dense = norm2d(input, sigma, center)
%function: generate 2d normal distribution
%@params:
%input: input size
%sigma: [sigmay, sigmax]
%center: [centerx, centery]
    gsize = size(input);
    [X1, X2] = meshgrid(1:gsize(1), 1:gsize(2));
    Sigma = zeros(size(sigma,2));
    for i = 1:size(sigma,2)
        Sigma(i,i) = sigma(i)^2;
    end
    dense = mvnpdf([X1(:) X2(:)], center, Sigma);
    dense = reshape(dense, gsize(2), gsize(1))';
end

function distance_matrix = distance(position_head, k)
%function: caculate the distance matrix
%@params:
%k: the k nearest neighbor
head_num = size(position_head, 1);
distance_matrix = zeros(head_num, head_num);
for i = 1:head_num
    for j = 1:head_num
        distance_matrix(i, j) = sum((position_head(i,:)-position_head(j,:)).^2);
    end
end
distance_matrix = sort(distance_matrix, 2);
distance_matrix = sqrt(distance_matrix(:, 2:k+1));
end
