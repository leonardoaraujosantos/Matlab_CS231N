%% Vectorized XOR example
% Implement a simple Neural network to handle the XOR problem with a 3
% layer perceptron (MLP) with 2 neurons on the input, 2 on the hidden layer
% and 1 on the output, we also use bias.
%
% <</home/leo/work/Matlab_CS231N/docs/imgs/XOR_NeuralNetwork.txt.png>>
%
%
% Every neuron on this ANN is connected to 3 weights, 2 weights coming from
% other neurons connections plus 1 connection with the bias, you can
% consider this as a 9 parmeter function

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
% http://outlace.com/Beginner-Tutorial-Backpropagation/
%

%% Define training dataset
% XOR input for x1 and x2
X = [0 0; 0 1; 1 0; 1 1];
% Desired output of XOR
Y_train = [0;1;1;0];

%% Define the sigmoid and dsigmoid
% Define the sigmoid (logistic) function and it's first derivative
sigmoid = @(x) 1.0 ./ ( 1.0 + exp(-x) );
dsigmoid = @(x) sigmoid(x) .* ( 1 - sigmoid(x) );

%% Initialization of meta parameters
% Learning coefficient
learnRate = 0.1; % Start to oscilate with 15
regularization = 0.00;
% Number of learning iterations
epochs = 6000;
smallStep = 0.0001;
% Minibatch = 4(train size) is Batch Gradient descent
% Minibatch = 1 is Stochastic Gradient descent
batchSize = 1; 
sizeTraining = size(X,1);

%% Cost function definition
% On this case we will use the Cross entropy cost(or loss) function, the
% idea of the loss function is to give a number to show how bad/good your
% current set of parameters are. Here the definition of good means that our
% ANN output matches the training dataset
J_vec = zeros(1,epochs);
lossFunction = CrossEntropy();

%% Weights random initialization
% This changes a lot the training performance! Some links:
%
% http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
%
% http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
%
% Here every weight row represent one neuron connection, ex weights(1,:)
% means all connections of the first neuron
%
% Put random number generator on state 0
%rand('state',0);
% Same thing but new syntax
rng(0,'v5uniform');
INIT_EPISLON = 0.8;
W1 = rand(2,3) * (2*INIT_EPISLON) - INIT_EPISLON;
W2 = rand(1,3) * (2*INIT_EPISLON) - INIT_EPISLON;
% Override manually to debug both vectorized and non vectorized
% implementation
%W1 = [-0.7690    0.6881   -0.2164; -0.0963    0.2379   -0.1385];
%W2 = [-0.1433   -0.4840   -0.6903];

% More neurons means more local minimas, which means easier training if you
% dont get stuck on a local minima
% Who is afraid of non convex loss-functions?
% http://videolectures.net/eml07_lecun_wia/
%W1 = rand(15,3) * (2*INIT_EPISLON) - INIT_EPISLON;
%W2 = rand(1,16) * (2*INIT_EPISLON) - INIT_EPISLON;

%% Training
initialIndex = 1;
for i = 1:epochs                
    numIterations = floor(sizeTraining / batchSize);
    complete_delta_1 = 0;
    complete_delta_2 = 0;
    h_vec = zeros(sizeTraining,1);
    
    % Shuffle the dataset
    ind = PartitionDataSet.getShuffledIndex(size(Y_train,1));
    X = X(ind,:);
    Y_train = Y_train(ind,:);
    
    for idxIter = 1:numIterations                
        % Extract a batch from the training
        if (numIterations ~= 1)
            batchFeatures = X(idxIter,:);
            batchLabels = Y_train(idxIter,:);
        else
            batchFeatures = X;
            batchLabels = Y_train;
        end
    
        %%% Backpropagation
        % Find the partial derivative of the cost function related to all
        % weights on the neural network (on our case 9 weights)
        
        %%% Forward pass
        % First Activation (Input-->Hidden)
        A1 = [ones(batchSize, 1) batchFeatures];
        Z2 = A1 * W1';
        A2 = sigmoid(Z2);
        
        % Second Activation (Hidden-->Output)
        A2=[ones(batchSize, 1) A2];
        Z3 = A2 * W2';
        A3 = sigmoid(Z3);
        h = A3;
        if numIterations == 4
            h_vec(idxIter) = h;
        else
            h_vec = h;
        end
        
        %%% Backward pass
        % For output layer: (Why different tutorials have differ here?)
        % delta = (1-actual output)*(desired output - actual output)
        %delta_out_layer = A3.*(1-A3).*(A3-Y_train); % Other
        %delta_out_layer = (Y_train-A3); % Andrew Ng
        delta_output = (A3-batchLabels); % Andrew Ng (Invert weight update signal)
        
        % For Hidden layer
        Z2=[ones(batchSize,1) Z2];
        delta_hidden=delta_output*W2.*dsigmoid(Z2);
        % Take out first column (bias column), to force the complete delta
        % to have the same size of it's respective weight
        delta_hidden=delta_hidden(:,2:end);
        
        % Calculate complete delta for every weight
        complete_delta_1 = complete_delta_1 + (delta_hidden'*A1);
        complete_delta_2 = complete_delta_2 + (delta_output'*A2);
        
        % Computing the partial derivatives with regularization, here we're
        % avoiding regularizing the bias term by substituting the first col of
        % weights with zeros
        p1 = ((regularization/sizeTraining)* [zeros(size(W1, 1), 1) W1(:, 2:end)]);
        p2 = ((regularization/sizeTraining)* [zeros(size(W2, 1), 1) W1(2, 2:end)]);
        D1 = (complete_delta_1 ./ sizeTraining) + p1;
        D2 = (complete_delta_2 ./ sizeTraining) + p2;

        %%% Weight Update
        % Gradient descent Update after all training set deltas are calculated
        % Increment or decrement depending on delta_output sign
        % Stochastic Gradient descent Update at every new input....
        % The stochastic gradient descent with luck converge faster ...
        % Increment or decrement depending on delta_output sign
        W1 = W1 - learnRate*(D1);
        W2 = W2 - learnRate*(D2);                
    end            
    %%% Calculate cost
    % After all calculations on the epoch calculate the cost function
    % Calculate Cost function output
    % Calculate p (used on regression)
    p = sum(sum(W1(:, 2:end).^2, 2))+sum(sum(W2(:, 2:end).^2, 2));
    % calculate J
    %J = sum(sum((-Y_train).*log(h) - (1-Y_train).*log(1-h), 2))/sizeTraining + regularization*p/(2*sizeTraining);
    J = lossFunction.getLoss(h_vec,Y_train) + regularization*p/(2*sizeTraining);
    J_vec(i) = J;
    %     % Break if error is already low
    %     if J < 0.08
    %         break;
    %     end
end

%% Plot some information
% Plot Prediction surface and Cost vs epoch curve
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
        % First Activation (Input-->Hidden)
        A1 = [ones(1, 1) test];
        Z2 = A1 * W1';
        A2 = sigmoid(Z2);
        
        % Second Activation (Hidden-->Output)
        A2=[ones(1, 1) A2];
        Z3 = A2 * W2';
        A3 = sigmoid(Z3);
        testOut(row, col) = A3;
        
        if isequal(test,[0 0]) || isequal(test,[0 1]) || isequal(test,[1 0]) || isequal(test,[1 1])
           fprintf('%d XOR %d ==> %d\n',test(1),test(2),round(A3));            
        end
    end
end
figure(2);
surf(X1, X2, testOut);
title('Prediction surface');

figure(1);
plot(J_vec);
title('Cost vs epochs');