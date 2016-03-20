%% Deep neural networks creation and simple tests (No training yet)

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
solver = SolverFactory.get(struct('type',SolverType.GradientDescent,'learningRate',0.011, 'numEpochs', 30));

nn = DeepNeuralNetwork(layers,solver);
% Training part is not to be tested yet so we put the weights manually...
layers.getLayer(1).weights = [-30; 20; 20];

Xt = [0 0; 0 1; 1 0; 1 1];
[maxscore, scores, predictTime] = nn.predict(Xt(1,:));
fprintf('%d AND %d = %d\n',Xt(1,1), Xt(1,2), scores);
[maxscore, scores, predictTime] = nn.predict(Xt(2,:));
fprintf('%d AND %d = %d\n',Xt(2,1), Xt(2,2), scores);
[maxscore, scores, predictTime] = nn.predict(Xt(3,:));
fprintf('%d AND %d = %d\n',Xt(3,1), Xt(3,2), scores);
[maxscore, scores, predictTime] = nn.predict(Xt(4,:));
fprintf('%d AND %d = %d\n',Xt(4,1), Xt(4,2), scores);

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
solver = SolverFactory.get(struct('type',SolverType.GradientDescent,'learningRate',0.011, 'numEpochs', 30));

nn = DeepNeuralNetwork(layers,solver);
% Training part is not to be tested yet so we put the weights manually...
layers.getLayer(1).weights = [-10; 20; 20];

Xt = [0 0; 0 1; 1 0; 1 1];
[maxscore, scores, predictTime] = nn.predict(Xt(1,:));
fprintf('%d OR %d = %d\n',Xt(1,1), Xt(1,2), scores);
[maxscore, scores, predictTime] = nn.predict(Xt(2,:));
fprintf('%d OR %d = %d\n',Xt(2,1), Xt(2,2), scores);
[maxscore, scores, predictTime] = nn.predict(Xt(3,:));
fprintf('%d OR %d = %d\n',Xt(3,1), Xt(3,2), scores);
[maxscore, scores, predictTime] = nn.predict(Xt(4,:));
fprintf('%d OR %d = %d\n',Xt(4,1), Xt(4,2), scores);

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
solver = SolverFactory.get(struct('type',SolverType.GradientDescent,'learningRate',0.011, 'numEpochs', 30));

nn = DeepNeuralNetwork(layers,solver);
% Training part is not to be tested yet so we put the weights manually...
layers.getLayer(1).weights = [-30 10; 20 -20; 20 -20];
layers.getLayer(2).weights = [-10; 20; 20];

Xt = [0 0; 0 1; 1 0; 1 1];
[maxscore, scores, predictTime] = nn.predict(Xt(1,:));
fprintf('%d XOR %d = %d\n',Xt(1,1), Xt(1,2), scores);
[maxscore, scores, predictTime] = nn.predict(Xt(2,:));
fprintf('%d XOR %d = %d\n',Xt(2,1), Xt(2,2), scores);
[maxscore, scores, predictTime] = nn.predict(Xt(3,:));
fprintf('%d XOR %d = %d\n',Xt(3,1), Xt(3,2), scores);
[maxscore, scores, predictTime] = nn.predict(Xt(4,:));
fprintf('%d XOR %d = %d\n',Xt(4,1), Xt(4,2), scores);


