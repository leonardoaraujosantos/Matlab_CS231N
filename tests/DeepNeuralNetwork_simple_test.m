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
% <</home/leo/work/Matlab_CS231N/docs/imgs/Simple_XOR_Coursera.PNG>>
%
% Perceptron test for AND
X = [0 0; 0 1; 1 0; 1 1];
Y = [ 0; 0; 0; 1];

layers = LayerContainer;
layers <= struct('type',LayerType.Input,'rows',2,'cols',1,'depth',1);
layers <= struct('type',LayerType.Output,'numClasses',1,'ActivationType',ActivationType.Sigmoid);
layers.showStructure();
solver = SolverFactory.get(struct('type',SolverType.GradientDescent,'learningRate',0.011, 'numEpochs', 30, 'RegularizationL1',0.00));

nn = DeepNeuralNetwork(layers,solver);
% Training part is not to be tested yet so we put the weights manually...
% First layer input has 2 neurons, and the output (1 neuron) so we're going
% 1x(2+1) [1 3] matrix
layers.getLayer(1).weights = [-30 20 20];

Xt = [0 0; 0 1; 1 0; 1 1];
[maxscore, scores, predictTime] = nn.predict(Xt(1,:));
fprintf('%d AND %d = %d\n',Xt(1,1), Xt(1,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(2,:));
fprintf('%d AND %d = %d\n',Xt(2,1), Xt(2,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(3,:));
fprintf('%d AND %d = %d\n',Xt(3,1), Xt(3,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(4,:));
fprintf('%d AND %d = %d\n',Xt(4,1), Xt(4,2), round(scores));

%% Test 2: Simple Perceptron OR
% Create the DNN strucutre and train with Gradient Descent
% Example
% https://www.coursera.org/learn/machine-learning/lecture/solUx/examples-and-intuitions-ii
% 
% <</home/leo/work/Matlab_CS231N/docs/imgs/Simple_XOR_Coursera.PNG>>
%
% Perceptron test for OR
layers = LayerContainer;
layers <= struct('type',LayerType.Input,'rows',2,'cols',1,'depth',1);
layers <= struct('type',LayerType.Output,'numClasses',1,'ActivationType',ActivationType.Sigmoid);
layers.showStructure();
solver = SolverFactory.get(struct('type',SolverType.GradientDescent,'learningRate',0.011, 'numEpochs', 30, 'RegularizationL1',0.00));

nn = DeepNeuralNetwork(layers,solver);
% Training part is not to be tested yet so we put the weights manually...
% First layer input has 2 neurons, and the output (1 neuron) so we're going
% 1x(2+1) [1 3] matrix
layers.getLayer(1).weights = [-10 20 20];

Xt = [0 0; 0 1; 1 0; 1 1];
[maxscore, scores, predictTime] = nn.predict(Xt(1,:));
fprintf('%d OR %d = %d\n',Xt(1,1), Xt(1,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(2,:));
fprintf('%d OR %d = %d\n',Xt(2,1), Xt(2,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(3,:));
fprintf('%d OR %d = %d\n',Xt(3,1), Xt(3,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(4,:));
fprintf('%d OR %d = %d\n',Xt(4,1), Xt(4,2), round(scores));

%% Test 3: Simple Multilayer layers creation
% Create the DNN strucutre and train with Gradient Descent
% Example
% https://www.coursera.org/learn/machine-learning/lecture/solUx/examples-and-intuitions-ii
% 
% <</home/leo/work/Matlab_CS231N/docs/imgs/Simple_XOR_Coursera.PNG>>
%
% Perceptron test for XNOR

layers = LayerContainer;
layers <= struct('type',LayerType.Input,'rows',2,'cols',1,'depth',1);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',2,'ActivationType',ActivationType.Sigmoid);
layers <= struct('type',LayerType.Output,'numClasses',1,'ActivationType',ActivationType.Sigmoid);
layers.showStructure();
solver = SolverFactory.get(struct('type',SolverType.GradientDescent,'learningRate',0.011, 'numEpochs', 30, 'RegularizationL1',0.00));

nn = DeepNeuralNetwork(layers,solver);
% Training part is not to be tested yet so we put the weights manually...
% Input layer has 2 neurons and the hidden layer more 2, so the 
% weight matrix will be 2x3 [2 3]
layers.getLayer(1).weights = [-30 20 20; 10 -20 -20]; 
% Second(hidden) layer has 2 neurons and the third(output) 1 neuron, so the
% weight matrix that will map layer 2 to 3 will have the format 1x3
layers.getLayer(2).weights = [-10 20 20];

Xt = [0 0; 0 1; 1 0; 1 1];
[maxscore, scores, predictTime] = nn.predict(Xt(1,:));
fprintf('%d XOR %d = %d\n',Xt(1,1), Xt(1,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(2,:));
fprintf('%d XOR %d = %d\n',Xt(2,1), Xt(2,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(3,:));
fprintf('%d XOR %d = %d\n',Xt(3,1), Xt(3,2), round(scores));
[maxscore, scores, predictTime] = nn.predict(Xt(4,:));
fprintf('%d XOR %d = %d\n',Xt(4,1), Xt(4,2), round(scores));


