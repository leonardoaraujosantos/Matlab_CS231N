
%[im, number_of_images] = load_thumbnails('C:\Users\Richard Burton\Work\Videos and datasets\Thumbnails\full_pos_test_set'); %load images
im{1} = img_batch(:,:,:,1);
im{2} = img_batch(:,:,:,2);
im{3} = img_batch(:,:,:,3);
im{4} = img_batch(:,:,:,4);
number_of_images = 4;

image_rgb_vectors_all = [];

for ii = 1: number_of_images %number of images to use

image_loaded = im2single(im{ii});

dimension1_image = size(image_loaded,1);
dimension2_image = size(image_loaded,2);

% 
image_loaded_x = im2col(image_loaded(:,:,1),[dimension1_image dimension2_image]); %reshape r channel to vector
image_loaded_y = im2col(image_loaded(:,:,2),[dimension1_image dimension2_image]); %reshape g channel to vector
image_loaded_z = im2col(image_loaded(:,:,3),[dimension1_image dimension2_image]); %reshape b channel to vector

% Delete later
R = image_loaded(:,:,1);
G = image_loaded(:,:,2);
B = image_loaded(:,:,3);
image_loaded_x = R(:);
image_loaded_y = G(:);
image_loaded_z = B(:);

image_rgb_vectors = [image_loaded_x, image_loaded_y, image_loaded_z]; %combine

image_rgb_vectors_all = vertcat(image_rgb_vectors_all , image_rgb_vectors); %add to other images 

end

[coeff,score,latent] = pca(image_rgb_vectors_all); 

alpha_1 =  randn/5;
alpha_2 =  randn/5;
alpha_3 =  randn/5;

alpha_ = [latent(1)*alpha_1;latent(2)*alpha_2;latent(3)*alpha_3]; 

pca_change = coeff*alpha_;

x1 = im2single(x1);
im_changed(:,:,1) = x1(:,:,1) + pca_change(1);
im_changed(:,:,2) = x1(:,:,2) + pca_change(2);
im_changed(:,:,3) = x1(:,:,3) + pca_change(3);

imagesc(im_changed)