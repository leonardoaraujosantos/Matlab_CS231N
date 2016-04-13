%% Deep neural networks creation and training

%% Test 1: Simple Multilayer layers XOR training
% Create the DNN strucutre and train with Gradient Descent
% Example
% https://www.coursera.org/learn/machine-learning/lecture/solUx/examples-and-intuitions-ii
% 
% <</home/leo/work/Matlab_CS231N/docs/imgs/Simple_XOR_Coursera.PNG>>
%
% MLP train for XOR
X = [0 0; 0 1; 1 0; 1 1];
Y = [ 0; 1; 1; 0];

layers = LayerContainer;
layers <= struct('type',LayerType.Input,'rows',2,'cols',1,'depth',1);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',2,'ActivationType',ActivationType.Sigmoid);
layers <= struct('type',LayerType.Output,'numClasses',1,'ActivationType',ActivationType.Sigmoid);
layers.showStructure();
solver = SolverFactory.get(struct('type',SolverType.GradientDescent,'learningRate',1, 'numEpochs', 2000, 'RegularizationL1',0.001));
% Force a batch size
solver.batch_size = 4;
nn = DeepNeuralNetwork(layers,solver);

% Train the neural network with the given solver (Type gradient descent)
nn.train(X, Y);

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
testInpx1 = [-1:0.1:1];
testInpx2 = [-1:0.1:1];
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

