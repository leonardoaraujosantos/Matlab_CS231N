%% Nice XOR example on matlab based on this tutorial (NonVectorized)
% http://matlabgeeks.com/tips-tutorials/neural-networks-a-multilayer-perceptron-in-matlab/
% http://www.holehouse.org/mlclass/09_Neural_Networks_Learning.html
% https://github.com/rasmusbergpalm/DeepLearnToolbox
% http://ufldl.stanford.edu/wiki/index.php/Neural_Network_Vectorization
% https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
% We have here a 3 layer network (1-Input, 2-Hidden, 3-Output)

% Sigmoid and dSigmoid functions
sigmoid = @(x) 1.0 ./ ( 1.0 + exp(-x) );
dsigmoid = @(x) sigmoid(x) .* ( 1 - sigmoid(x) );

num_layers = 3;
% XOR input for x1 and x2
X = [0 0; 0 1; 1 0; 1 1];
% Desired output of XOR
Y_train = [0;1;1;0];
% Initialize the bias
bias = [1 1 1];
% Learning coefficient
learnRate = 2;
% Number of learning iterations
epochs = 20000;
regularization = 0.00;
% Calculate weights randomly using seed.
% We have 5 neurons and 3 bias
INIT_EPISLON = 0.8;
weights = rand(3,3) * (2*INIT_EPISLON) - INIT_EPISLON;

J_vec = zeros(1,epochs);
CrossEntrInst = CrossEntropy();

%% Training
for i = 1:epochs
    outNN = zeros(4,1);
    sizeTraining = length (X(:,1));
    complete_delta_out = 0;
    complete_delta_2_2 = 0;
    complete_delta_2_1 = 0;
    for j = 1:sizeTraining
        %% Forward pass
        % First Neuron hidden layer
        Z_a1 = bias(1,1)*weights(1,1) + X(j,1)*weights(1,2) + X(j,2)*weights(1,3);
        x2(1) = sigmoid(Z_a1);
        a1 = sigmoid(Z_a1);
        
        % Second Neuron hidden layer
        Z_a2 = bias(1,2)*weights(2,1) + X(j,1)*weights(2,2)+ X(j,2)*weights(2,3);
        x2(2) = sigmoid(Z_a2);
        a2 = sigmoid(Z_a2);
        
        % Third neuron output layer
        Z3 = bias(1,3)*weights(3,1) + a1*weights(3,2) + a2*weights(3,3);
        outNN(j) = sigmoid(Z3);
        
        %% Backpropagation
        % Find the partial derivative of the cost function related to all
        % weights on the neural network (on our case 9 weights)
        
        % For output layer:
        % delta(wi) = xi*delta,
        % delta = (1-actual output)*(desired output - actual output)
        %delta_out_layer = (1-outNN(j))*(Y_train(j)-outNN(j)); % Other
        delta_out_layer = (outNN(j)-Y_train(j)); % Andrew Ng
        
        % For Hidden layer
        delta2_2 = (weights(3,3)*delta_out_layer) * dsigmoid(Z_a2);
        delta2_1 = (weights(3,2)*delta_out_layer) * dsigmoid(Z_a1);
        
        % Calculate complete delta for every weight (Testing....)
        complete_delta_2_2 = complete_delta_2_2 + (a2*delta_out_layer);
        complete_delta_2_1 = complete_delta_2_1 + (a1*delta_out_layer);
        
        % Stochastic Gradient descent??? Update at every new input....
        % Update weights
        for k = 1:num_layers
            if k == 1 % Bias cases
                weights(1,k) = weights(1,k) - learnRate*bias(1,1)*delta2_1;
                weights(2,k) = weights(2,k) - learnRate*bias(1,2)*delta2_2;
                weights(3,k) = weights(3,k) - learnRate*bias(1,3)*delta_out_layer;
            else % When k=2 or 3 input cases to neurons
                weights(1,k) = weights(1,k) - learnRate*X(j,1)*delta2_1;
                weights(2,k) = weights(2,k) - learnRate*X(j,2)*delta2_2;
                weights(3,k) = weights(3,k) - learnRate*x2(k-1)*delta_out_layer;
            end
        end
    end
    % After all calculations on the epoch calculate the cost function
    % Calculate Cost function output
    % Calculate p (used on regression)
    p = sum(sum(weights.^2, 2)); % Not ready yet for non-vectorized regularization
    J = CrossEntrInst.getLoss(outNN,Y_train) + regularization*p/(2*sizeTraining);
    %J = CrossEntrInst.getLoss(outNN,Y_train);
    J_vec(i) = J;
    
    % Break if error is already low
    if J < 0.08
        J_vec = J_vec(1:i);
        break;
    end
end

fprintf('Outputs\n');
disp(round(outNN));

%% Plot some information
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
        % First Neuron hidden layer
        Z_a1 = bias(1,1)*weights(1,1) + test(1)*weights(1,2) + test(2)*weights(1,3);
        x2(1) = sigmoid(Z_a1);
        a1 = sigmoid(Z_a1);
        
        % Second Neuron hidden layer
        Z_a2 = bias(1,2)*weights(2,1) + test(1)*weights(2,2)+ test(2)*weights(2,3);
        x2(2) = sigmoid(Z_a2);
        a2 = sigmoid(Z_a2);
        
        % Third neuron output layer
        Z3 = bias(1,3)*weights(3,1) + a1*weights(3,2) + a2*weights(3,3);
        testOut(row, col) = sigmoid(Z3);
    end
end
figure(2);
surf(X1, X2, testOut);
title('Prediction surface');

figure(1);
plot(J_vec);
title('Cost vs epochs');