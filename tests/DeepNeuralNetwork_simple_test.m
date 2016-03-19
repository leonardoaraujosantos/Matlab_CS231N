%% Deep neural networks creation and simple tests

%% Test 1: Simple Perceptron layers creation
% Perceptron test for OR
X = [0 0; 0 1; 1 0; 1 1];
Y = [ 0; 1; 1; 1];

% Create the DNN strucutre and train with Gradient Descent
layers = LayerContainer;
layers <= struct('type',LayerType.Input,'rows',1,'cols',2,'depth',1);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',4,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.OutputSoftMax,'numClasses',1);
layers.showStructure();
solver = SolverFactory.get(struct('type',SolverType.GradientDescent,'learningRate',0.011, 'numEpochs', 30));
nn = DeepNeuralNetwork(layers,solver);

trainTime = nn.train(X,Y);
fprintf('Train time: %d\n',trainTime);
[maxscore, scores, predictTime] = nn.predict([1 0]);
fprintf('Predict time: %d\n',predictTime);
fprintf('maxScore: %d\n',maxscore);
fprintf('Scores: %d\n',scores);


