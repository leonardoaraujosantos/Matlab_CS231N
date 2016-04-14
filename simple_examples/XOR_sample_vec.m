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
learnRate = 2; % Start to oscilate with 15
regularization = 0.00;
% Number of learning iterations
epochs = 2000;
smallStep = 0.0001;

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
for i = 1:epochs
    
    %%% Numeric estimation
    % Used to debug backpropagation, it will calculate numerically the
    % partial derivative of the cost function related to every
    % parameter, but much slower than backpropagation, just to have an
    % idea every time you have to calculate the partial derivative
    % related to one specific weight, you need to calculate the cost
    % function to every item on the training twice (J1,J2), and to
    % calculate the cost function you need to do foward propagation on
    % the whole network and this must be done every time the gradient
    % descent change the weight on the local minima direction
    %
    % <</home/leo/work/Matlab_CS231N/docs/imgs/GradientChecking.PNG>>
    %
    %
    % <</home/leo/work/Matlab_CS231N/docs/imgs/GradientChecking2.png>>
    %
    %
    % http://www.coursera.org/learn/machine-learning/lecture/Y3s6r/gradient-checking
    %
    Thetas = [W1(:) ; W2(:)];
    numgrad = zeros(size(Thetas));
    perturb = zeros(size(Thetas));
    hidden_layer_size = 2;
    input_layer_size = 2;
    output_layer_size = 1;
    for p = 1:numel(Thetas)
        % Set perturbation vector
        perturb(p) = smallStep;
        ThetasLoss1 = Thetas - perturb;
        ThetasLoss2 = Thetas + perturb;
        
        % Loss 1
        % Reshape Theta on the W1,W2 format
        nW1 = reshape(ThetasLoss1(1:hidden_layer_size * ...
            (input_layer_size + 1)), hidden_layer_size, ...
            (input_layer_size + 1));
        nW2 = reshape(ThetasLoss1((1 + (hidden_layer_size * ...
            (input_layer_size + 1))):end), output_layer_size, ...
            (hidden_layer_size + 1));
        
        % Forward prop
        A1 = [ones(sizeTraining, 1) X];
        Z2 = A1 * nW1';
        A2 = sigmoid(Z2);
        A2 = [ones(sizeTraining, 1) A2];
        Z3 = A2 * nW2';
        A3 = sigmoid(Z3);
        h1 = A3;
        
        % Loss 2
        % Reshape Theta on the W1,W2 format
        nW1 = reshape(ThetasLoss2(1:hidden_layer_size * ...
            (input_layer_size + 1)), hidden_layer_size, ...
            (input_layer_size + 1));
        nW2 = reshape(ThetasLoss2((1 + (hidden_layer_size * ...
            (input_layer_size + 1))):end), output_layer_size, ...
            (hidden_layer_size + 1));
        % Forward prop
        A1 = [ones(sizeTraining, 1) X];
        Z2 = A1 * nW1';
        A2 = sigmoid(Z2);
        A2 = [ones(sizeTraining, 1) A2];
        Z3 = A2 * nW2';
        A3 = sigmoid(Z3);
        h2 = A3;
        
        % Calculate both losses...
        loss1 = CrossEntrInst.getLoss(h1,Y_train);
        loss2 = CrossEntrInst.getLoss(h2,Y_train);
        
        % Compute Numerical Gradient
        numgrad(p) = (loss2 - loss1) / (2*smallStep);
        perturb(p) = 0;
    end
    numDeltaW1 = reshape(numgrad(1:hidden_layer_size * ...
        (input_layer_size + 1)), hidden_layer_size, ...
        (input_layer_size + 1));
    numDeltaW2 = reshape(numgrad((1 + (hidden_layer_size * ...
        (input_layer_size + 1))):end), output_layer_size, ...
        (hidden_layer_size + 1));
    
    %%% Backpropagation
    % Find the partial derivative of the cost function related to all
    % weights on the neural network (on our case 9 weights)
    
    %%% Forward pass
    % First Activation (Input-->Hidden)
    A1 = [ones(sizeTraining, 1) X];
    Z2 = A1 * W1';
    A2 = sigmoid(Z2);
    
    % Second Activation (Hidden-->Output)
    A2=[ones(sizeTraining, 1) A2];
    Z3 = A2 * W2';
    A3 = sigmoid(Z3);
    h = A3;
    
    %%% Backward pass
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
    
    % Calculate complete delta for every weight
    complete_delta_1 = (delta_hidden'*A1);
    complete_delta_2 = (delta_output'*A2);
    
    % Computing the partial derivatives with regularization, here we're
    % avoiding regularizing the bias term by substituting the first col of
    % weights with zeros
    p1 = ((regularization/sizeTraining)* [zeros(size(W1, 1), 1) W1(:, 2:end)]);
    p2 = ((regularization/sizeTraining)* [zeros(size(W2, 1), 1) W1(2, 2:end)]);
    D1 = (complete_delta_1 ./ sizeTraining) + p1;
    D2 = (complete_delta_2 ./ sizeTraining) + p2;
    
    %%% Check backpropagation
    % Compare backpropagation partial derivatives with numerical gradients
    % We should do this check just few times. This is because calculating
    % the numeric gradient every time is heavy.
    errorBackPropD1 = sum(sum(abs(D1 - numDeltaW1)));
    errorBackPropD2 = sum(sum(abs(D2 - numDeltaW2)));
    % Stop if backpropagation error bigger than 0.0001
    if (errorBackPropD1 > 0.001) || (errorBackPropD2 > 0.001)
        fprintf('Backpropagation error %d %d\n',errorBackPropD1,errorBackPropD2);
        pause;
    end
    
    %%% Weight Update
    % Gradient descent Update after all training set deltas are calculated
    % Increment or decrement depending on delta_output sign
    % Stochastic Gradient descent Update at every new input....
    % The stochastic gradient descent with luck converge faster ...
    % Increment or decrement depending on delta_output sign
    W1 = W1 - learnRate*(D1);
    W2 = W2 - learnRate*(D2);
    
    %%% Calculate cost
    % After all calculations on the epoch calculate the cost function
    % Calculate Cost function output
    % Calculate p (used on regression)
    p = sum(sum(W1(:, 2:end).^2, 2))+sum(sum(W2(:, 2:end).^2, 2));
    % calculate J
    %J = sum(sum((-Y_train).*log(h) - (1-Y_train).*log(1-h), 2))/sizeTraining + regularization*p/(2*sizeTraining);
    J = CrossEntrInst.getLoss(h,Y_train) + regularization*p/(2*sizeTraining);
    J_vec(i) = J;
    %     % Break if error is already low
    %     if J < 0.08
    %         break;
    %     end
end

fprintf('Outputs\n');
disp(round(A3));

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
    end
end
figure(2);
surf(X1, X2, testOut);
title('Prediction surface');

figure(1);
plot(J_vec);
title('Cost vs epochs');