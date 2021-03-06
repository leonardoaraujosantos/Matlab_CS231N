%% Test 1: Check NearestNeighbor one sample
load fisheriris
X = meas;
Y = species;

% Shuffle data and divide by 2 (test/train)
ind = randperm(size(X, 1));
train_X = X(ind(1:ceil(size(X, 1)/2)),:);
train_Y = Y(ind(1:ceil(size(Y, 1)/2)),:);

test_X = X(ind(1:ceil(size(X, 1)/2))+1:end,:);
test_Y = Y(ind(1:ceil(size(Y, 1)/2))+1:end,:);

% Load Matlab version if knn(1) classifier (default euclidian L2 distance)
Mdl = fitcknn(train_X,train_Y,'NumNeighbors',1);

% Load Our version
classifier = NearestNeighbor();
classifier.train(train_X,train_Y);

% Select one sample to predict fifith example after trainning (bottom half)
X_test = test_X(5,:);
Y_test = test_Y(5,:);
predictWith = cell2mat(Y_test);
fprintf('Predicting with a image of type %s\n',predictWith);

% Predict with our classifier and with matlab knn(1) classifer
[maxscore, ~, ~] = classifier.predict(X_test,2);
maxscore_matlab = predict(Mdl,X_test);
assert (strcmp(cell2mat(maxscore), cell2mat(maxscore_matlab)) == 1);

%% Test 2: Test global accuracy
load fisheriris
X = meas;
Y = species;

% Shuffle data and divide by 2 (test/train)
rng(1); % Fix a fix seed for random values
ind = randperm(size(X, 1));
X = X(ind,:);
Y = Y(ind,:);
numTrain = floor(size(X,1)/2);
train_X = X(1:numTrain,:);
train_Y = Y(1:numTrain,:);

test_X = X(numTrain+1:end,:);
test_Y = Y(numTrain+1:end,:);
numTest = size(test_X,1);

% Load Matlab version if knn(1) classifier (default euclidian L2 distance)
Mdl = fitcknn(train_X,train_Y,'NumNeighbors',1);

% Load Our version
classifier = NearestNeighbor();
classifier.train(train_X,train_Y);

countPredictCorrectMatlab = 0;
countPredictCorrectNN = 0;

% Iterate on all test samples
for idx=1:numTest
    % Select sample
    X_test = test_X(idx,:);
    Y_test = test_Y(idx,:);
    
    [maxscore, ~, ~] = classifier.predict(X_test,2);
    maxscore_matlab = predict(Mdl,X_test);
    
    % Check correct classification
    if strcmp(cell2mat(maxscore), cell2mat(Y_test)) == 1
        countPredictCorrectNN = countPredictCorrectNN + 1;
    end    
    if strcmp(cell2mat(maxscore_matlab), cell2mat(Y_test)) == 1
        countPredictCorrectMatlab = countPredictCorrectMatlab + 1;
    end
end

% Calculate and compare accuracy
accuracy = (countPredictCorrectNN)/numTest;
accuracyMatlab = (countPredictCorrectMatlab)/numTest;
fprintf('Accuracy custom nn: %d matlab knn(1): %d\n',accuracy,accuracyMatlab);
assert (accuracy == accuracyMatlab);