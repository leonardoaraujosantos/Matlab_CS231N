%% Nice XOR example on matlab based on this tutorial
% http://matlabgeeks.com/tips-tutorials/neural-networks-a-multilayer-perceptron-in-matlab/
% http://www.holehouse.org/mlclass/09_Neural_Networks_Learning.html
% https://github.com/rasmusbergpalm/DeepLearnToolbox
% We have here a 3 layer network (1-Input, 2-Hidden, 3-Output)

% Sigmoid and dSigmoid functions
sigmoid = @(x) 1.0 ./ ( 1.0 + exp(-x) );
dsigmoid = @(x) sigmoid(x) .* ( 1 - sigmoid(x) );

num_layers = 3;
% XOR input for x1 and x2
X = [0 0; 0 1; 1 0; 1 1];
% Desired output of XOR
Y_train = [0;1;1;0];
% Learning coefficient
learnRate = 0.1;
% Number of learning iterations
epochs = 10000;
% Calculate weights randomly using seed.
INIT_EPISLON = 0.1;
W1 = rand(2,3) * (2*INIT_EPISLON) - INIT_EPISLON;
W2 = rand(1,3) * (2*INIT_EPISLON) - INIT_EPISLON;

%% Training
for i = 1:epochs
    sizeTraining = length (X(:,1));
    complete_delta_1 = 0;
    complete_delta_2 = 0;
    for j = 1:sizeTraining
        %% Forward pass
        % First Activation (Input-->Hidden)
        A1 = [ones(sizeTraining, 1) X];
        Z2 = A1 * W1';
        A2 = sigmoid(Z2);
        
        % Second Activation (Hidden-->Output)
        A2=[ones(sizeTraining, 1) A2];
        Z3 = A2 * W2';
        A3 = sigmoid(Z3);
        
        %% Backpropagation
        % Find the partial derivative of the cost function related to all
        % weights on the neural network (on our case 9 weights)
        
        % For output layer: (Why different tutorials have differ here?)
        % delta = (1-actual output)*(desired output - actual output)
        %delta_out_layer = (1-A3).*(Y_train-A3); % Other
        %delta_out_layer = (Y_train-A3); % Andrew Ng
        delta_output = (A3-Y_train); % Andrew Ng (Invert weight update signal)
        
        % For Hidden layer
        Z2=[ones(sizeTraining,1) Z2];
        delta_hidden=delta_output*W2.*dsigmoid(Z2);
        
        % Calculate complete delta for every weight (Testing....)
        delta_hidden=delta_hidden(:,2:end);
        complete_delta_1 = complete_delta_1 + (delta_hidden'*A1);
        complete_delta_2 = complete_delta_2 + (delta_output'*A2);
        
        % Stochastic Gradient descent Update at every new input....
        % The stochastic gradient descent with luck converge faster ...
        % Increment or decrement depending on delta_output sign
        W1 = W1 - learnRate*complete_delta_1;
        W2 = W2 - learnRate*complete_delta_2;
    end
    % Gradient descent Update after all training set deltas are calculated
    % Increment or decrement depending on delta_output sign
    %W1 = W1 - learnRate*complete_delta_1;
    %W2 = W2 - learnRate*complete_delta_2;
end

fprintf('Outputs\n');
disp(round(A3));

fprintf('\nW1\n');
disp(W1);

fprintf('\nW2\n');
disp(W2);

%% Plot some information
% Plot cost function vs epoch

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
        % First Activation (Input-->Hidden)
        A1 = [ones(1, 1) test];
        Z2 = A1 * W1';
        A2 = sigmoid(Z2);
        
        % Second Activation (Hidden-->Output)
        A2=[ones(1, 1) A2];
        Z3 = A2 * W2';
        A3 = sigmoid(Z3);
        testOut(row, col) = A3;
    end
end
figure(2);
surf(X1, X2, testOut);
title('Prediction surface');