%% Deep neural networks creation and simple tests

%% Test 1: Simple Perceptron layers creation
% Create the DNN strucutre and train with Gradient Descent
% Example
% https://www.coursera.org/learn/machine-learning/lecture/solUx/examples-and-intuitions-ii
% 
% <</home/leo/work/Matlab_CS231N/docs/imgs/Simple_XOR_Coursera.PNG>>
%
% Perceptron test for XOR
X = [0 0; 0 1; 1 0; 1 1];
Y = [ 0; 1; 1; 0];

layers = LayerContainer;
layers <= struct('type',LayerType.Input,'rows',2,'cols',1,'depth',1);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',2,'ActivationType',ActivationType.Sigmoid);
layers <= struct('type',LayerType.OutputSoftMax,'numClasses',1);
layers.showStructure();
solver = SolverFactory.get(struct('type',SolverType.GradientDescent,'learningRate',0.011, 'numEpochs', 30));

nn = DeepNeuralNetwork(layers,solver);
% Training part is not to be tested yet so we put the weights manually...
layers.getLayer(1).weights = [-30 20 20; 10 -20 -20];
layers.getLayer(2).weights = [-10; 20; 20];

%trainTime = nn.train(X,Y);
%fprintf('Train time: %d\n',trainTime);
[maxscore, scores, predictTime] = nn.predict([1 0]);
fprintf('Predict time: %d\n',predictTime);
fprintf('maxScore: %d\n',maxscore);
fprintf('Scores: %d\n',scores);


