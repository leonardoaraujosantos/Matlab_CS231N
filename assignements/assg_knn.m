%% K-Nearest Neighbor (knn) exercise

%% Load Cifar-10 dataset
clear all; close all; clc;
load('datasets/batches.meta.mat');
tb_b1 = loadCifar_10('datasets/data_batch_1.mat','datasets/batches.meta.mat');
tb_b2 = loadCifar_10('datasets/data_batch_2.mat','datasets/batches.meta.mat');
tb_b3 = loadCifar_10('datasets/data_batch_3.mat','datasets/batches.meta.mat');
tb_b4 = loadCifar_10('datasets/data_batch_4.mat','datasets/batches.meta.mat');
tb_b5 = loadCifar_10('datasets/data_batch_5.mat','datasets/batches.meta.mat');
table_cifar_10_test = loadCifar_10('datasets/test_batch.mat','datasets/batches.meta.mat');
table_cifar_10_train = [tb_b1; tb_b2; tb_b3; tb_b4; tb_b5];
clear tb_b1 tb_b2 tb_b3 tb_b4 tb_b5;

%% Take out a little bit of data to make it run faster
table_cifar_10_train = table_cifar_10_train(1:5000,:);
table_cifar_10_test = table_cifar_10_test(1:500,:);

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
imgTestIdx = 1; % With 99 predicted right
classifier = NearestNeighbor();
X_train = double(cell2mat(table_cifar_10_train.X));
Y_train = double(table_cifar_10_train.Y);
timeToTrain = classifier.train(X_train,Y_train);
fprintf('time to train: %d\n',timeToTrain);

X_test = double(cell2mat(table_cifar_10_test.X));
Y_test = double(table_cifar_10_test.Y);
predictWith = cell2mat(table_cifar_10_test.Desc(imgTestIdx));
fprintf('Predicting with a image of type %s\n',predictWith);

[maxscore, scores, timeToPredict] = classifier.predict(X_test(imgTestIdx,:),1);
predictedDesc = label_names{maxscore+1};
fprintf('time to predict: %d desc:%s(%d)\n',timeToPredict,predictedDesc,maxscore);
fprintf('Correct answer should be %s\n',label_names{Y_test(imgTestIdx)+1});

% Try with matlab command
Mdl = fitcknn(X_train,Y_train,'NumNeighbors',1);
maxscore_matlab = predict(Mdl,X_test(imgTestIdx,:));
predictedDesc = label_names{maxscore_matlab+1};
fprintf('Using matlab knn(k=1): desc:%s(%d)\n',predictedDesc,maxscore_matlab);

% Calculate accuracy over the test set
countCorrect = 0;
countCorrectMatlab = 0;
for idxTest=1:500    
    [maxscore, ~, ~] = classifier.predict(X_test(idxTest,:),2);    
    maxscore_matlab = predict(Mdl,X_test(idxTest,:));
    if (Y_test(idxTest) == maxscore)
        countCorrect = countCorrect + 1;
    end    
    if (Y_test(idxTest) == maxscore_matlab)
        countCorrectMatlab = countCorrectMatlab + 1;
    end
end
accuracy = (countCorrect)/500;
accuracyMatlab = (countCorrectMatlab)/500;
fprintf('Accuracy custom nn: %d matlab knn(1): %d\n',accuracy,accuracyMatlab);

%% Test with KNearest Neighbor
imgTestIdx = 1; % With 99 predicted right
K = 8;
classifier = KNearestNeighbor();
X_train = double(cell2mat(table_cifar_10_train.X));
Y_train = double(table_cifar_10_train.Y);
timeToTrain = classifier.train(X_train,Y_train);
fprintf('time to train: %d\n',timeToTrain);

X_test = double(cell2mat(table_cifar_10_test.X));
Y_test = double(table_cifar_10_test.Y);
predictWith = cell2mat(table_cifar_10_test.Desc(imgTestIdx));
fprintf('Predicting with a image of type %s\n',predictWith);

[maxscore, scores, timeToPredict] = classifier.predict(X_test(imgTestIdx,:),1,K);
predictedDesc = label_names{maxscore+1};
fprintf('time to predict: %d desc:%s(%d)\n',timeToPredict,predictedDesc,maxscore);
fprintf('Correct answer should be %s\n',label_names{Y_test(imgTestIdx)+1});

% Try with matlab command
Mdl = fitcknn(X_train,Y_train,'NumNeighbors',K);
maxscore_matlab = predict(Mdl,X_test(imgTestIdx,:));
predictedDesc = label_names{maxscore_matlab+1};
fprintf('Using matlab knn(k=7): desc:%s(%d)\n',predictedDesc,maxscore_matlab);

% Calculate accuracy over the test set
countCorrect = 0;
countCorrectMatlab = 0;
for idxTest=1:500    
    [maxscore, ~, ~] = classifier.predict(X_test(idxTest,:),2,K);    
    maxscore_matlab = predict(Mdl,X_test(idxTest,:));
    if (Y_test(idxTest) == maxscore)
        countCorrect = countCorrect + 1;
    end    
    if (Y_test(idxTest) == maxscore_matlab)
        countCorrectMatlab = countCorrectMatlab + 1;
    end
end
accuracy = (countCorrect)/500;
accuracyMatlab = (countCorrectMatlab)/500;
fprintf('Accuracy custom knn(7): %d matlab knn(7): %d\n',accuracy,accuracyMatlab);

%% Cross validation
num_folds = 5;
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100];





