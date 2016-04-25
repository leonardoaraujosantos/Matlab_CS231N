%% Deep neural networks creation and training

%% Test 1: Simple Multilayer layers XOR training (Sigmoid)
% Create the DNN strucutre and train with Gradient Descent
% Example
% https://www.coursera.org/learn/machine-learning/lecture/solUx/examples-and-intuitions-ii
% 
% <</home/leo/work/Matlab_CS231N/docs/imgs/Simple_XOR_Coursera.PNG>>
%
% MLP train for XOR
X = [0 0; 0 1; 1 0; 1 1];
Y = [ 0; 1; 1; 0];

% Reset random number generator state
rng(0,'v5uniform');

layers = LayerContainer;
layers <= struct('type',LayerType.Input,'rows',2,'cols',1,'depth',1);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',2,'ActivationType',ActivationType.Sigmoid);
layers <= struct('type',LayerType.Output,'numClasses',1,'ActivationType',ActivationType.Sigmoid);
layers.showStructure();
solver = SolverFactory.get(struct('type',SolverType.GradientDescent,'learningRate',2, 'numEpochs', 2000, 'RegularizationL1',0.01));
% Force a batch size
solver.batch_size = 4;
% Get a loss function object to be used on the training
lossFunction = CrossEntropy();
nn = DeepNeuralNetwork(layers,solver,lossFunction);
%nn.layers.getLayer(1).weights = [-0.7690    0.6881   -0.2164; -0.0963    0.2379   -0.1385];
%nn.layers.getLayer(2).weights = [-0.1433   -0.4840   -0.6903];

% Train the neural network with the given solver (Type gradient descent)
timeTrain = nn.train(X, Y);
fprintf('Time to train %2.1d seconds\n',timeTrain);

% Weights for XNOR (Comment this, if you train automatically)
%layers.getLayer(1).weights = [-30 20 20; 10 -20 -20]; 
%layers.getLayer(2).weights = [-10 20 20];

Xt = [0 0; 0 1; 1 0; 1 1];
[maxscore, scores, predictTime] = nn.predict(Xt(1,:));
fprintf('%d XOR %d = %d\n',Xt(1,1), Xt(1,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(2,:));
fprintf('%d XOR %d = %d\n',Xt(2,1), Xt(2,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(3,:));
fprintf('%d XOR %d = %d\n',Xt(3,1), Xt(3,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(4,:));
fprintf('%d XOR %d = %d\n',Xt(4,1), Xt(4,2), round(scores));

% Plot Prediction surface
testInpx1 = [-0.5:0.1:1.5];
testInpx2 = [-0.5:0.1:1.5];
[X1, X2] = meshgrid(testInpx1, testInpx2);
testOutRows = size(X1, 1);
testOutCols = size(X1, 2);
testOut = zeros(testOutRows, testOutCols);
for row = [1:testOutRows]
    for col = [1:testOutCols]
        test = [X1(row, col), X2(row, col)];
        %% Forward pass     
        [~,A3,~] = nn.predict(test);
        testOut(row, col) = A3;
    end
end
figure(2);
surf(X1, X2, testOut);
title('Prediction surface');
figure(1);
plot(nn.lossVector);
title('Cost vs epochs');


%% Test 2: Simple Multilayer layers XOR training (Relu)
% Create the DNN strucutre and train with Gradient Descent
% Example
% https://www.coursera.org/learn/machine-learning/lecture/solUx/examples-and-intuitions-ii
% 
% <</home/leo/work/Matlab_CS231N/docs/imgs/Simple_XOR_Coursera.PNG>>
%
% MLP train for XOR
X = [0 0; 0 1; 1 0; 1 1];
Y = [ 0; 1; 1; 0];

% Reset random number generator state
rng(0,'v5uniform');

layers = LayerContainer;
layers <= struct('type',LayerType.Input,'rows',2,'cols',1,'depth',1);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',3,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.Output,'numClasses',1,'ActivationType',ActivationType.Sigmoid);
layers.showStructure();
solver = SolverFactory.get(struct('type',SolverType.GradientDescent,'learningRate',2, 'numEpochs', 2000, 'RegularizationL1',0.01));
% Force a batch size
solver.batch_size = 4;
% Get a loss function object to be used on the training
lossFunction = CrossEntropy();
nn = DeepNeuralNetwork(layers,solver,lossFunction);
%nn.layers.getLayer(1).weights = [-0.7690    0.6881   -0.2164; -0.0963    0.2379   -0.1385];
%nn.layers.getLayer(2).weights = [-0.1433   -0.4840   -0.6903];

% Train the neural network with the given solver (Type gradient descent)
timeTrain = nn.train(X, Y);
fprintf('Time to train %2.1d seconds\n',timeTrain);

% Weights for XNOR (Comment this, if you train automatically)
%layers.getLayer(1).weights = [-30 20 20; 10 -20 -20]; 
%layers.getLayer(2).weights = [-10 20 20];

Xt = [0 0; 0 1; 1 0; 1 1];
[maxscore, scores, predictTime] = nn.predict(Xt(1,:));
fprintf('%d XOR %d = %d\n',Xt(1,1), Xt(1,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(2,:));
fprintf('%d XOR %d = %d\n',Xt(2,1), Xt(2,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(3,:));
fprintf('%d XOR %d = %d\n',Xt(3,1), Xt(3,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(4,:));
fprintf('%d XOR %d = %d\n',Xt(4,1), Xt(4,2), round(scores));

% Plot Prediction surface
testInpx1 = [-0.5:0.1:1.5];
testInpx2 = [-0.5:0.1:1.5];
[X1, X2] = meshgrid(testInpx1, testInpx2);
testOutRows = size(X1, 1);
testOutCols = size(X1, 2);
testOut = zeros(testOutRows, testOutCols);
for row = [1:testOutRows]
    for col = [1:testOutCols]
        test = [X1(row, col), X2(row, col)];
        %% Forward pass     
        [~,A3,~] = nn.predict(test);
        testOut(row, col) = A3;
    end
end
figure(2);
surf(X1, X2, testOut);
title('Prediction surface');
figure(1);
plot(nn.lossVector);
title('Cost vs epochs');


%% Test 3: Simple Multilayer layers XOR training (Deep)
% Create the DNN strucutre and train with Gradient Descent
% Example
% https://www.coursera.org/learn/machine-learning/lecture/solUx/examples-and-intuitions-ii
% 
% <</home/leo/work/Matlab_CS231N/docs/imgs/Simple_XOR_Coursera.PNG>>
%
% MLP train for XOR
X = [0 0; 0 1; 1 0; 1 1];
Y = [ 0; 1; 1; 0];

% Reset random number generator state
rng(0,'v5uniform');

layers = LayerContainer;
layers <= struct('type',LayerType.Input,'rows',2,'cols',1,'depth',1);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',10,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',10,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',10,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',10,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.Output,'numClasses',1,'ActivationType',ActivationType.Sigmoid);
layers.showStructure();
solver = SolverFactory.get(struct('type',SolverType.GradientDescent,'learningRate',0.1, 'numEpochs', 2000, 'RegularizationL1',0.01));
% Force a batch size
solver.batch_size = 4;
% Get a loss function object to be used on the training
lossFunction = CrossEntropy();
nn = DeepNeuralNetwork(layers,solver,lossFunction);
%nn.layers.getLayer(1).weights = [-0.7690    0.6881   -0.2164; -0.0963    0.2379   -0.1385];
%nn.layers.getLayer(2).weights = [-0.1433   -0.4840   -0.6903];

% Train the neural network with the given solver (Type gradient descent)
timeTrain = nn.train(X, Y);
fprintf('Time to train %2.1d seconds\n',timeTrain);

% Weights for XNOR (Comment this, if you train automatically)
%layers.getLayer(1).weights = [-30 20 20; 10 -20 -20]; 
%layers.getLayer(2).weights = [-10 20 20];

Xt = [0 0; 0 1; 1 0; 1 1];
[maxscore, scores, predictTime] = nn.predict(Xt(1,:));
fprintf('%d XOR %d = %d\n',Xt(1,1), Xt(1,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(2,:));
fprintf('%d XOR %d = %d\n',Xt(2,1), Xt(2,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(3,:));
fprintf('%d XOR %d = %d\n',Xt(3,1), Xt(3,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(4,:));
fprintf('%d XOR %d = %d\n',Xt(4,1), Xt(4,2), round(scores));

% Plot Prediction surface
testInpx1 = [-0.5:0.1:1.5];
testInpx2 = [-0.5:0.1:1.5];
[X1, X2] = meshgrid(testInpx1, testInpx2);
testOutRows = size(X1, 1);
testOutCols = size(X1, 2);
testOut = zeros(testOutRows, testOutCols);
for row = [1:testOutRows]
    for col = [1:testOutCols]
        test = [X1(row, col), X2(row, col)];
        %% Forward pass     
        [~,A3,~] = nn.predict(test);
        testOut(row, col) = A3;
    end
end
figure(2);
surf(X1, X2, testOut);
title('Prediction surface');
figure(1);
plot(nn.lossVector);
title('Cost vs epochs');

%% Test 4: Simple Multilayer layers XOR training (Deep, linear last layer)
% Create the DNN strucutre and train with Gradient Descent
% Example
% https://www.coursera.org/learn/machine-learning/lecture/solUx/examples-and-intuitions-ii
% 
% <</home/leo/work/Matlab_CS231N/docs/imgs/Simple_XOR_Coursera.PNG>>
%
% MLP train for XOR
X = [0 0; 0 1; 1 0; 1 1];
Y = [ 0; 1; 1; 0];

% Reset random number generator state
rng(0,'v5uniform');

layers = LayerContainer;
layers <= struct('type',LayerType.Input,'rows',2,'cols',1,'depth',1);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',10,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',10,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',10,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',10,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.Output,'numClasses',1,'ActivationType',ActivationType.Linear);
layers.showStructure();
solver = SolverFactory.get(struct('type',SolverType.GradientDescent,'learningRate',0.1, 'numEpochs', 2000, 'RegularizationL1',0.01));
% Force a batch size
solver.batch_size = 4;
% Get a loss function object to be used on the training
lossFunction = CrossEntropy();
nn = DeepNeuralNetwork(layers,solver,lossFunction);
%nn.layers.getLayer(1).weights = [-0.7690    0.6881   -0.2164; -0.0963    0.2379   -0.1385];
%nn.layers.getLayer(2).weights = [-0.1433   -0.4840   -0.6903];

% Train the neural network with the given solver (Type gradient descent)
timeTrain = nn.train(X, Y);
fprintf('Time to train %2.1d seconds\n',timeTrain);

% Weights for XNOR (Comment this, if you train automatically)
%layers.getLayer(1).weights = [-30 20 20; 10 -20 -20]; 
%layers.getLayer(2).weights = [-10 20 20];

Xt = [0 0; 0 1; 1 0; 1 1];
[maxscore, scores, predictTime] = nn.predict(Xt(1,:));
fprintf('%d XOR %d = %d\n',Xt(1,1), Xt(1,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(2,:));
fprintf('%d XOR %d = %d\n',Xt(2,1), Xt(2,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(3,:));
fprintf('%d XOR %d = %d\n',Xt(3,1), Xt(3,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(4,:));
fprintf('%d XOR %d = %d\n',Xt(4,1), Xt(4,2), round(scores));

% Plot Prediction surface
testInpx1 = [-0.5:0.1:1.5];
testInpx2 = [-0.5:0.1:1.5];
[X1, X2] = meshgrid(testInpx1, testInpx2);
testOutRows = size(X1, 1);
testOutCols = size(X1, 2);
testOut = zeros(testOutRows, testOutCols);
for row = [1:testOutRows]
    for col = [1:testOutCols]
        test = [X1(row, col), X2(row, col)];
        %% Forward pass     
        [~,A3,~] = nn.predict(test);
        testOut(row, col) = A3;
    end
end
figure(2);
surf(X1, X2, testOut);
title('Prediction surface');
figure(1);
plot(nn.lossVector);
title('Cost vs epochs');

%% Test 4: Train on MNIST dataset
% Load dataset of 5000 images (20x20)
load mnist_matlab_ready

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
display_MNIST_Data(X(sel, :));
trainSize = size(X, 1);
y = oneHot(y);

% Get 10 elements to test
sel = randperm(size(X, 1));
sel = sel(1:100);
X_test = X(sel, :);
y_test = y(sel, :);
testSize = size(X_test, 1);

% Reset random number generator state
rng(0,'v5uniform');

% Prepare network 5 layers 1-Input, 3-Hidden, 1-Output
layers = LayerContainer;
layers <= struct('type',LayerType.Input,'rows',20*20,'cols',1,'depth',1);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',100,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',100,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',100,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',100,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.Output,'numClasses',10,'ActivationType',ActivationType.Linear);
layers.showStructure();
solver = SolverFactory.get(struct('type',SolverType.GradientDescent,'learningRate',0.01, 'numEpochs', 6000, 'RegularizationL1',0.01));
% Force a batch size
solver.batch_size = trainSize;
% Get a loss function object to be used on the training
lossFunction = CrossEntropy();
nn = DeepNeuralNetwork(layers,solver,lossFunction);

% Train the neural network with the given solver (Type gradient descent)
timeTrain = nn.train(X, y);
fprintf('Time to train %2.1d seconds\n',timeTrain);

% Test
figure(2);
display_MNIST_Data(X(sel, :));
title('Images on validation');
errorCount = 0;
for idxTest=1:testSize
    [idx_MaxScore,scores,~] = nn.predict(X_test(idxTest,:));
    [~, trained_out] = max(y_test(idxTest,:));
    if idx_MaxScore == 10
        idx_MaxScore = 0;
    end
    if trained_out == 10
        trained_out = 0;
    end
    if trained_out ~= idx_MaxScore
        errorCount = errorCount + 1;
    end
    fprintf('Predicted %d and should be %d\n',idx_MaxScore,trained_out);    
end
errorPercentage = (errorCount*100) / testSize;
fprintf('Accuracy is %d percent \n',(100-errorPercentage));

figure(1);
plot(nn.lossVector);
title('Cost vs epochs');
