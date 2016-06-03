%% Deep neural networks creation and training

%% Test 2: Simple Multilayer layers XOR training (Relu)
% Create the DNN strucutre and train with Gradient Descent
% Example
% https://www.coursera.org/learn/machine-learning/lecture/solUx/examples-and-intuitions-ii
% 
% <<../../docs/imgs/Simple_XOR_Coursera.PNG>>
%
% MLP train for XOR
X = [0 0; 0 1; 1 0; 1 1];
% Put X on format (width)x(height)x(channel)x(batchsize)
X = reshape(X',[1,2,1,4]);
Y = [ 0; 1; 1; 0];

% Reset random number generator state
rng(0,'v5uniform');

% Multi-layer perceptron test for XOR
% Thinking on neural network structure we would have the first input layer
% with 2 neurons, a hidden layer with more 2 neurons and an output layer
% with 1 neuron
numClasses = 1;
layers = LayerContainer;
layers <= struct('type',LayerType.Input,'rows',1,'cols',2,'depth',1);
layers <= struct('type',LayerType.InnerProduct, 'numOutputs',2);
layers <= struct('type',LayerType.Sigmoid);
layers <= struct('type',LayerType.InnerProduct, 'numOutputs',numClasses);
layers <= struct('type',LayerType.Sigmoid);
layers.showStructure();

layers.showStructure();

% Get a loss function object to be used on the training
lossFunction = SoftMaxLoss();
myModel = DeepLearningModel(layers);

% Set the model loss function
myModel.lossFunction = lossFunction;

optimizer = Optimizer();
optimizer.configs.learning_rate = 0.1;
optimizer.configs.momentum = 0.9;
solver = Solver(myModel, optimizer, {X, Y});
solver.batchSize = 4;
solver.num_epochs = 2000;
solver.train();

Xt = [0 0; 0 1; 1 0; 1 1];
[maxscore, scores, predictTime] = myModel.loss(Xt(1,:));
fprintf('%d XOR %d = %d\n',Xt(1,1), Xt(1,2), round(scores));
[maxscore, scores, predictTime] = myModel.loss(Xt(2,:));
fprintf('%d XOR %d = %d\n',Xt(2,1), Xt(2,2), round(scores));
[maxscore, scores, predictTime] = myModel.loss(Xt(3,:));
fprintf('%d XOR %d = %d\n',Xt(3,1), Xt(3,2), round(scores));
[maxscore, scores, predictTime] = myModel.loss(Xt(4,:));
fprintf('%d XOR %d = %d\n',Xt(4,1), Xt(4,2), round(scores));

