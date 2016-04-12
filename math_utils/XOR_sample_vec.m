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
learnRate = 1; % Start to oscilate with 15
regularization = 0.001;
% Number of learning iterations
epochs = 2000;
% Calculate weights randomly using seed.
INIT_EPISLON = 0.8;
W1 = rand(2,3) * (2*INIT_EPISLON) - INIT_EPISLON;
W2 = rand(1,3) * (2*INIT_EPISLON) - INIT_EPISLON;

% More neurons means more local minimas, which means easier training if you
% dont get stuck on a local minima
% Who is afraid of non convex loss-functions?
% http://videolectures.net/eml07_lecun_wia/
%W1 = rand(5,3) * (2*INIT_EPISLON) - INIT_EPISLON;
%W2 = rand(1,6) * (2*INIT_EPISLON) - INIT_EPISLON;

J_vec = zeros(1,epochs);

%% Training
for i = 1:epochs
    sizeTraining = length (X(:,1));
    complete_delta_1 = 0;
    complete_delta_2 = 0;
    
    %% Forward pass
    % First Activation (Input-->Hidden)
    A1 = [ones(sizeTraining, 1) X];
    Z2 = A1 * W1';
    A2 = sigmoid(Z2);
    
    % Second Activation (Hidden-->Output)
    A2=[ones(sizeTraining, 1) A2];
    Z3 = A2 * W2';
    A3 = sigmoid(Z3);
    h = A3;
    
    %% Backpropagation
    % Find the partial derivative of the cost function related to all
    % weights on the neural network (on our case 9 weights)
    
    % For output layer: (Why different tutorials have differ here?)
    % delta = (1-actual output)*(desired output - actual output)
    %delta_out_layer = A3.*(1-A3).*(A3-Y_train); % Other
    %delta_out_layer = (Y_train-A3); % Andrew Ng
    delta_output = (A3-Y_train); % Andrew Ng (Invert weight update signal)
    
    % For Hidden layer
    Z2=[ones(sizeTraining,1) Z2];
    delta_hidden=delta_output*W2.*dsigmoid(Z2);
    % Take out first column (bias column), to force the complete delta
    % to have the same size of it's respective weight
    delta_hidden=delta_hidden(:,2:end);
    
    % Calculate complete delta for every weight (Testing....)
    complete_delta_1 = complete_delta_1 + (delta_hidden'*A1);
    complete_delta_2 = complete_delta_2 + (delta_output'*A2);
    
    % Computing the partial derivatives with regularization, here we're
    % avoiding regularizing the bias term by substituting the first col of
    % weights with zeros
    D1 = ((1/sizeTraining) * complete_delta_1) + ((regularization/sizeTraining)* [zeros(size(W1, 1), 1) W1(:, 2:end)]);
    D2 = ((1/sizeTraining) * complete_delta_2) + ((regularization/sizeTraining)* [zeros(size(W2, 1), 1) W2(:, 2:end)]);
    
    % Gradient descent Update after all training set deltas are calculated
    % Increment or decrement depending on delta_output sign
    % Stochastic Gradient descent Update at every new input....
    % The stochastic gradient descent with luck converge faster ...
    % Increment or decrement depending on delta_output sign
    W1 = W1 - learnRate*(D1);
    W2 = W2 - learnRate*(D2);
    
    % After all calculations on the epoch calculate the cost function
    % Calculate Cost function output
    p = sum(sum(W1(:, 2:end).^2, 2))+sum(sum(W2(:, 2:end).^2, 2));
    % calculate J
    J = sum(sum((-Y_train).*log(h) - (1-Y_train).*log(1-h), 2))/sizeTraining + regularization*p/(2*sizeTraining);
    J_vec(i) = J;
    % Break if error is already low
%     if J < 0.08        
%         break;
%     end
end

fprintf('Outputs\n');
disp(round(A3));

%fprintf('\nW1\n');
%disp(W1);

%fprintf('\nW2\n');
%disp(W2);

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

figure(1);
plot(J_vec);
title('Cost vs epochs');