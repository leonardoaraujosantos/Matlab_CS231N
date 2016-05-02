%% Load data
clear all
load mnist_oficial

%% Create model
model = TwoLayerNet(784,50,10);
optimizer = Optimizer();
optimizer.configs.learning_rate = 0.05;
optimizer.configs.momentum = 0.1;

%% Create and configure solver
solver = Solver(model, optimizer);
solver.X_train = input_train;
solver.Y_train = output_train;
solver.X_val = input_test;
solver.Y_val = output_test;
solver.batchSize = 64;
solver.num_epochs = 150;
solver.learn_rate_decay = 0.95;
solver.verbose = 1;

%% Train
solver.train

%% Plot loss
figure(1);
plot(solver.validationAccuracyVector);
title('Accuracy plot');
figure(2);
plot(solver.lossVector);
title('Loss plot');