%% K-Nearest Neighbor (knn) exercise

%% Load Cifar-10 dataset
load('datasets/batches.meta.mat');
tb_b1 = loadCifar_10('datasets/data_batch_1.mat','datasets/batches.meta.mat');
tb_b2 = loadCifar_10('datasets/data_batch_2.mat','datasets/batches.meta.mat');
tb_b3 = loadCifar_10('datasets/data_batch_3.mat','datasets/batches.meta.mat');
tb_b4 = loadCifar_10('datasets/data_batch_4.mat','datasets/batches.meta.mat');
tb_b5 = loadCifar_10('datasets/data_batch_5.mat','datasets/batches.meta.mat');
table_cifar_10_test = loadCifar_10('datasets/test_batch.mat','datasets/batches.meta.mat');
table_cifar_10_train = [tb_b1; tb_b2; tb_b3; tb_b4; tb_b5];
clear tb_b1 tb_b2 tb_b3 tb_b4 tb_b5;

%% Vizualize some classes 
classes =  label_names;
num_classes = length(classes);
samples_per_class = 7;
plotImgCounter = 1;
for idxClass=1:num_classes
    class_desc = classes{idxClass};
    % Select all the rows where ID==idxClass
    rowsClass = table_cifar_10_train.Y==idxClass-1;
    imgClass = table_cifar_10_train.Image(rowsClass);
    for idxImg=1:samples_per_class
        img = imgClass{idxImg};
        axis off;
        subplot(num_classes, samples_per_class, plotImgCounter);
        imshow(img);
        plotImgCounter = plotImgCounter + 1;
    end        
end

%% Test with Nearest Neighbor
figure(2);
imgTestIdx = 99;
classifier = NearestNeighbor();
X_train = cell2mat(table_cifar_10_train.X);
Y_train = table_cifar_10_train.Y;
timeToTrain = classifier.train(X_train,Y_train);
fprintf('time to train: %d\n',timeToTrain);

X_test = cell2mat(table_cifar_10_train.X);
predictWith = cell2mat(table_cifar_10_train.Desc(imgTestIdx));
imshow(cell2mat(table_cifar_10_train.Image(imgTestIdx)));
fprintf('Predicting with a image of type %s\n',predictWith);

[maxscore, scores, timeToPredict] = classifier.predict(X_test(imgTestIdx,:),1);
predictedDesc = label_names{maxscore+1};
fprintf('time to predict: %d desc:%s\n',timeToPredict,predictedDesc);


