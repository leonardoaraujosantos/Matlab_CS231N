%% Deep neural networks creation and simple tests (No training yet)
% We calculate the weigh matrix of a particular layer by taking the number
% of neurons of the next layer times the number of neurons of the current
% layer + 1
% Ex layer_1(2 neurons) layer_2(4 neurons) so the weight_1_2 (Weight from
% layer 1 that maps to layer 2) will have the format 4x3

%% Test 1: Simple Perceptron AND
% Create the DNN strucutre and train with Gradient Descent
% Example
% https://www.coursera.org/learn/machine-learning/lecture/solUx/examples-and-intuitions-ii
% 
% <<../../docs/imgs/Simple_XOR_Coursera.PNG>>
%
% Perceptron test for AND
X = [0 0; 0 1; 1 0; 1 1];
Y = [ 0; 0; 0; 1];
numClasses = 1;

% This is a structure of a perceptron, so it can handle only linear
% separable problems
layers = LayerContainer;
layers <= struct('type',LayerType.Input,'rows',1,'cols',2,'depth',1);
layers <= struct('type',LayerType.InnerProduct, 'numOutputs',numClasses);
layers <= struct('type',LayerType.Sigmoid);
layers.showStructure();

nn = DeepLearningModel(layers);
% Training part is not to be tested yet so we put the weights manually...
% Weights on the innerProduct (Fully connected layer)
layers.getLayer(2).biasWeights = -30;
layers.getLayer(2).weights = [20; 20];

Xt = [0 0; 0 1; 1 0; 1 1];
[maxscore, scores, predictTime] = nn.loss(Xt(1,:));
fprintf('%d AND %d = %d\n',Xt(1,1), Xt(1,2), round(scores));
[maxscore, scores, predictTime] = nn.loss(Xt(2,:));
fprintf('%d AND %d = %d\n',Xt(2,1), Xt(2,2), round(scores));
[maxscore, scores, predictTime] = nn.loss(Xt(3,:));
fprintf('%d AND %d = %d\n',Xt(3,1), Xt(3,2), round(scores));
[maxscore, scores, predictTime] = nn.loss(Xt(4,:));
fprintf('%d AND %d = %d\n',Xt(4,1), Xt(4,2), round(scores));

%% Test 2: Simple Perceptron OR
% Create the DNN strucutre and train with Gradient Descent
% Example
% https://www.coursera.org/learn/machine-learning/lecture/solUx/examples-and-intuitions-ii
% 
% <<../../docs/imgs/Simple_XOR_Coursera.PNG>>
%
% Perceptron test for OR
% This is a structure of a perceptron, so it can handle only linear
% separable problems
layers = LayerContainer;
layers <= struct('type',LayerType.Input,'rows',1,'cols',2,'depth',1);
layers <= struct('type',LayerType.InnerProduct, 'numOutputs',numClasses);
layers <= struct('type',LayerType.Sigmoid);
layers.showStructure();

nn = DeepLearningModel(layers);
% Training part is not to be tested yet so we put the weights manually...
% Weights on the innerProduct (Fully connected layer)
layers.getLayer(2).biasWeights = -10;
layers.getLayer(2).weights = [20; 20];

Xt = [0 0; 0 1; 1 0; 1 1];
[maxscore, scores, predictTime] = nn.loss(Xt(1,:));
fprintf('%d OR %d = %d\n',Xt(1,1), Xt(1,2), round(scores));
[maxscore, scores, predictTime] = nn.loss(Xt(2,:));
fprintf('%d OR %d = %d\n',Xt(2,1), Xt(2,2), round(scores));
[maxscore, scores, predictTime] = nn.loss(Xt(3,:));
fprintf('%d OR %d = %d\n',Xt(3,1), Xt(3,2), round(scores));
[maxscore, scores, predictTime] = nn.loss(Xt(4,:));
fprintf('%d OR %d = %d\n',Xt(4,1), Xt(4,2), round(scores));

%% Test 3: Simple Multilayer layers creation
% Create the DNN strucutre and train with Gradient Descent
% Example
% https://www.coursera.org/learn/machine-learning/lecture/solUx/examples-and-intuitions-ii
% 
% <<../../docs/imgs/Simple_XOR_Coursera.PNG>>
%
% Multi-layer perceptron test for XNOR
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

nn = DeepLearningModel(layers);
% Training part is not to be tested yet so we put the weights manually...
% Weights on the innerProduct (Fully connected layer)
% On previous neural network model
%layers.getLayer(1).weights = [-30 20 20; 10 -20 -20]; 
%layers.getLayer(2).weights = [-10 20 20];

% First fully connected layer
layers.getLayer(2).biasWeights = [-30 10];
layers.getLayer(2).weights = [20 -20; 20 -20];

% Second fully connected layer
layers.getLayer(4).biasWeights = -10;
layers.getLayer(4).weights = [20; 20];

Xt = [0 0; 0 1; 1 0; 1 1];
[maxscore, scores, predictTime] = nn.loss(Xt(1,:));
fprintf('%d XNOR %d = %d\n',Xt(1,1), Xt(1,2), round(scores));
[maxscore, scores, predictTime] = nn.loss(Xt(2,:));
fprintf('%d XNOR %d = %d\n',Xt(2,1), Xt(2,2), round(scores));
[maxscore, scores, predictTime] = nn.loss(Xt(3,:));
fprintf('%d XNOR %d = %d\n',Xt(3,1), Xt(3,2), round(scores));
[maxscore, scores, predictTime] = nn.loss(Xt(4,:));
fprintf('%d XNOR %d = %d\n',Xt(4,1), Xt(4,2), round(scores));


