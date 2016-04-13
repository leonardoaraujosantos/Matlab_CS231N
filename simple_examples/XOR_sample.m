%% Non-Vectorized XOR example
% Implement a simple Neural network to handle the XOR problem

%% Tutorials
% http://matlabgeeks.com/tips-tutorials/neural-networks-a-multilayer-perceptron-in-matlab/
%
% http://www.holehouse.org/mlclass/09_Neural_Networks_Learning.html
%
% http://github.com/rasmusbergpalm/DeepLearnToolbox
%
% http://ufldl.stanford.edu/wiki/index.php/Neural_Network_Vectorization
%
% http://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
% 
% http://github.com/stephencwelch/Neural-Networks-Demystified
%
% We have here a 3 layer network (1-Input, 2-Hidden, 3-Output)

%% Define the sigmoid and dsigmoid
% Define the sigmoid (logistic) function and it's first derivative
sigmoid = @(x) 1.0 ./ ( 1.0 + exp(-x) );
dsigmoid = @(x) sigmoid(x) .* ( 1 - sigmoid(x) );

%% Initialization
num_layers = 3;
% XOR input for x1 and x2
X = [0 0; 0 1; 1 0; 1 1];
% Desired output of XOR
Y_train = [0;1;1;0];
% Initialize the bias
bias = 1;
% Learning coefficient
learnRate = 2;
% Number of learning iterations
epochs = 20000;
regularization = 0.00;
% Calculate weights randomly using seed.
% We have 5 neurons and 3 bias
INIT_EPISLON = 0.8;
J_vec = zeros(1,epochs);
CrossEntrInst = CrossEntropy();

%% Weights random initialization
% Every neuron on this ANN is connected to 3 weights, 2 weights coming from
% other neurons connections plus 1 connection with the bias
% Here every weight row represent one neuron connection, ex weights(1,:)
% means all connections of the first neuron
weights = rand(3,3) * (2*INIT_EPISLON) - INIT_EPISLON;
% Override manually to debug both vectorized and non vectorized
% implementation
weights = [-0.7690 0.6881 -0.2164; -0.0963 0.2379 -0.1385; -0.1433 -0.4840 -0.6903];
disp(weights);

%% Training
for i = 1:epochs
    outNN = zeros(4,1);
    sizeTraining = length (X(:,1));
    accDelta1 = 0;accDelta2 = 0;accDelta3 = 0;accDelta4 = 0;accDelta5 = 0;
    accDelta6 = 0;
    
    for j = 1:sizeTraining
        %%% Forward pass
        % First Neuron hidden layer
        Z_a1 = bias*weights(1,1) + X(j,1)*weights(1,2) + X(j,2)*weights(1,3);
        x2(1) = sigmoid(Z_a1);
        a1 = sigmoid(Z_a1);
        
        % Second Neuron hidden layer
        Z_a2 = bias*weights(2,1) + X(j,1)*weights(2,2)+ X(j,2)*weights(2,3);
        x2(2) = sigmoid(Z_a2);
        a2 = sigmoid(Z_a2);
        
        % Third neuron output layer
        Z3 = bias*weights(3,1) + a1*weights(3,2) + a2*weights(3,3);
        outNN(j) = sigmoid(Z3);
        
        %%% Backpropagation        
        % Find the partial derivative of the cost function related to all
        % weights on the neural network (on our case 9 weights)
        %
        % http://www.youtube.com/watch?v=GlcnxUlrtek
        %
        % https://www.youtube.com/watch?v=5u0jaA3qAGk
        %        
        % For output layer:
        % delta(wi) = xi*delta,
        % delta = (1-actual output)*(desired output - actual output)
        %delta_out_layer = (1-outNN(j))*(Y_train(j)-outNN(j)); % Other
        delta_out_layer = (outNN(j)-Y_train(j)); % Andrew Ng
        
        % For Hidden layer
        delta2_2 = (weights(3,3)*delta_out_layer) * dsigmoid(Z_a2);
        delta2_1 = (weights(3,2)*delta_out_layer) * dsigmoid(Z_a1);
        
        % Accumulate deltas (Used to Gradient descent)
        accDelta1 = accDelta1 + bias*delta2_1;
        accDelta2 = accDelta2 + bias*delta2_2;
        accDelta3 = accDelta3 + bias*delta_out_layer;
        accDelta4 = accDelta4 + X(j,1)*delta2_1;
        accDelta5 = accDelta5 + X(j,2)*delta2_2;
        accDelta6 = accDelta6 + bias*delta2_1;        
        
        % Stochastic Gradient descent Update at every new input....
        % Update weights
        % Delta: CurrentLayerActivations * delta(nextLayer)
        for k = 1:num_layers
            if k == 1 % Bias cases
                weights(1,k) = weights(1,k) - learnRate*(bias*delta2_1);
                weights(2,k) = weights(2,k) - learnRate*(bias*delta2_2);
                weights(3,k) = weights(3,k) - learnRate*bias*(delta_out_layer);
            else % When k=2 or 3 input cases to neurons
                % Update with learnRate * Activations * smallDelta
                weights(1,k) = weights(1,k) - learnRate*(X(j,1)*delta2_1);
                weights(2,k) = weights(2,k) - learnRate*(X(j,2)*delta2_2);
                weights(3,k) = weights(3,k) - learnRate*(x2(k-1)*delta_out_layer);
            end
        end
    end
%     % Gradient descent Update after passing on all elements of training
%     % Update weights
%     % Delta: CurrentLayerActivations * delta(nextLayer)
%     for k = 1:num_layers
%         if k == 1 % Bias cases
%             weights(1,k) = weights(1,k) - learnRate*accDelta1;
%             weights(2,k) = weights(2,k) - learnRate*accDelta2;
%             weights(3,k) = weights(3,k) - learnRate*accDelta3;
%         else % When k=2 or 3 input cases to neurons
%             % Update with learnRate * Activations * smallDelta
%             weights(1,k) = weights(1,k) - learnRate*accDelta4;
%             weights(2,k) = weights(2,k) - learnRate*accDelta5;
%             weights(3,k) = weights(3,k) - learnRate*(x2(k-1)*delta_out_layer);
%         end
%     end
    
    %%% Cost function calculation
    % After all calculations on the epoch calculate the cost function
    % Calculate Cost function output
    % Calculate p (used on regression)
    p = sum(sum(weights.^2, 2)); % Not ready yet for non-vectorized regularization
    J = CrossEntrInst.getLoss(outNN,Y_train) + regularization*p/(2*sizeTraining);
    %J = CrossEntrInst.getLoss(outNN,Y_train);
    J_vec(i) = J;
    
    %%% Early stop
    % Break if error is already low
    if J < 0.08
        J_vec = J_vec(1:i);
        break;
    end
end

%% Outputs
fprintf('XOR ANN trained weights\n');
disp(weights);

fprintf('XOR ANN trained outputs\n');
disp(round(outNN));

%% Plots
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
        
        % Forward pass
        % First Neuron hidden layer
        Z_a1 = bias*weights(1,1) + test(1)*weights(1,2) + test(2)*weights(1,3);
        x2(1) = sigmoid(Z_a1);
        a1 = sigmoid(Z_a1);
        
        % Second Neuron hidden layer
        Z_a2 = bias*weights(2,1) + test(1)*weights(2,2)+ test(2)*weights(2,3);
        x2(2) = sigmoid(Z_a2);
        a2 = sigmoid(Z_a2);
        
        % Third neuron output layer
        Z3 = bias*weights(3,1) + a1*weights(3,2) + a2*weights(3,3);
        testOut(row, col) = sigmoid(Z3);
    end
end
figure(2);
surf(X1, X2, testOut);
title('Prediction surface');

figure(1);
plot(J_vec);
title('Cost vs epochs');