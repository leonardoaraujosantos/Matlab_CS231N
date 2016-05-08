%% Non-Vectorized XOR example
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
num_layers = 3;
% Initialize the bias
bias = 1;
% Learning coefficient
learnRate = 2;
% Number of learning iterations
epochs = 2000;
regularization = 0.00;
smallStep = 0.0001;
sizeTraining = length (X(:,1));

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
% Override manually to debug both vectorized and non vectorized
% implementation
weights = [ -0.7690 0.6881 -0.2164; ...
            -0.0963 0.2379 -0.1385; ...
            -0.1433 -0.4840 -0.6903];
%weights = rand(3,3) * (2*INIT_EPISLON) - INIT_EPISLON;
disp('Initital Weights');
disp(weights);

%% Creating symbolically the Neural network
% As said earlier we can consider this network as a 9 parameter function
syms X1 X2 theta1 theta2 theta3 theta4 theta5 theta6 theta7 theta8 theta9 p_act
g = symfun(1/(1+exp(-p_act)),p_act);
H_X_thetas = symfun( g( theta7 + (g(theta1+(X1*theta2)+(X2*theta3))*theta8) + (g(theta4+(X1*theta5)+(X2*theta6))*theta9) ) ,[X1,X2,theta1,theta2,theta3,theta4,theta5,theta6,theta7,theta8,theta9]);
H_mat_X_thetas = matlabFunction(H_X_thetas);
pretty(H_X_thetas);

% Compare to see of the first forward propagation with X(0,0) gives the
% same result
thetas = weights';
firstVal = H_mat_X_thetas(0,0,thetas(1),thetas(2),thetas(3),thetas(4)...
    ,thetas(5),thetas(6),thetas(7),thetas(8),thetas(9));

%% Creating symbolically the Loss function
%LossSymbolic = (-1/sizeTraining) * 

%% Use Calculus to get the partial derivatives of the cost function
% Well... Calculus on the computer...
partDeriv_1 = matlabFunction(diff(H_X_thetas,theta1));
partDeriv_2 = matlabFunction(diff(H_X_thetas,theta2));
partDeriv_3 = matlabFunction(diff(H_X_thetas,theta3));
partDeriv_4 = matlabFunction(diff(H_X_thetas,theta4));
partDeriv_5 = matlabFunction(diff(H_X_thetas,theta5));
partDeriv_6 = matlabFunction(diff(H_X_thetas,theta6));
partDeriv_7 = matlabFunction(diff(H_X_thetas,theta7));
partDeriv_8 = matlabFunction(diff(H_X_thetas,theta8));
partDeriv_9 = matlabFunction(diff(H_X_thetas,theta9));

%% Training
% The objective here is to minimize the cost function, or find the set of
% weights that make the output of the neural network be the same as the
% training dataset
for i = 1:epochs
    outNN = zeros(4,1);
    accDelta = zeros(3,3);
    
    for j = 1:sizeTraining
        %%% Do a numerical gradient estimation
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
        approxDerivative = zeros(size(weights));
        for idxWeight = 1:numel(weights)
            weights2 = weights;
            outNNApprox = zeros(4,1);
            for idxT = 1:sizeTraining
                % Forward pass to calculate J1
                weights2(idxWeight) = weights2(idxWeight) + smallStep;
                Z_a1 = bias*weights2(1,1) + ...
                    X(idxT,1)*weights2(1,2) + X(idxT,2)*weights2(1,3);
                a1 = sigmoid(Z_a1);
                
                % Second Neuron hidden layer
                Z_a2 = bias*weights2(2,1) + ...
                    X(idxT,1)*weights2(2,2)+ X(idxT,2)*weights2(2,3);
                a2 = sigmoid(Z_a2);
                
                % Third neuron output layer
                Z3 = bias*weights2(3,1) + a1*weights2(3,2) + a2*weights2(3,3);
                a3 = sigmoid(Z3);
                outNNApprox(idxT) = a3;
            end
            
            % Calculate cost J1, should include regularization???
            p = sum(sum(weights2.^2, 2));
            J1 = lossFunction.getLoss(outNNApprox,Y_train) + regularization*p/(2*sizeTraining);
            
            % Forward pass to calculate J2
            weights2 = weights;
            outNNApprox = zeros(4,1);
            
            for idxT = 1:sizeTraining
                % Forward pass to calculate J1
                weights2(idxWeight) = weights2(idxWeight) - smallStep;
                Z_a1 = bias*weights2(1,1) + ...
                    X(idxT,1)*weights2(1,2) + X(idxT,2)*weights2(1,3);
                a1 = sigmoid(Z_a1);
                
                % Second Neuron hidden layer
                Z_a2 = bias*weights2(2,1) + ...
                    X(idxT,1)*weights2(2,2)+ X(idxT,2)*weights2(2,3);
                a2 = sigmoid(Z_a2);
                
                % Third neuron output layer
                Z3 = bias*weights2(3,1) + a1*weights2(3,2) + a2*weights2(3,3);
                a3 = sigmoid(Z3);
                outNNApprox(idxT) = a3;
            end
            
            % Calculate cost J2, should include regularization???
            p = sum(sum(weights2.^2, 2));
            J2 = lossFunction.getLoss(outNNApprox,Y_train) + regularization*p/(2*sizeTraining);
            
            % All this to aproximate the partial derivative
            approxDerivative(idxWeight) = (J1-J2)/(2*smallStep);
        end
        
        %%% Backpropagation
        % Find the partial derivative of the cost function related to all
        % weights on the neural network (on our case 9 weights)
        %
        % http://www.youtube.com/watch?v=GlcnxUlrtek
        %
        % http://www.youtube.com/watch?v=5u0jaA3qAGk
        %
        % http://www.coursera.org/learn/machine-learning/lecture/du981/backpropagation-intuition
        %
        %%% Forward pass
        % First Neuron hidden layer
        Z_a1 = bias*weights(1,1) + ...
            X(j,1)*weights(1,2) + X(j,2)*weights(1,3);
        x2(1) = sigmoid(Z_a1);
        a1 = sigmoid(Z_a1);
        
        % Second Neuron hidden layer
        Z_a2 = bias*weights(2,1) + ...
            X(j,1)*weights(2,2)+ X(j,2)*weights(2,3);
        x2(2) = sigmoid(Z_a2);
        a2 = sigmoid(Z_a2);
        
        % Third neuron output layer
        Z3 = bias*weights(3,1) + a1*weights(3,2) + a2*weights(3,3);
        outNN(j) = sigmoid(Z3);
        
        %%% Calculate deltas
        % For output layer:
        % delta(wi) = xi*delta,
        % delta = (1-actual output)*(desired output - actual output)
        %delta_out = (1-outNN(j))*(Y_train(j)-outNN(j)); % Other
        %delta_out = (outNN(j) - Y_train(j))* dsigmoid(outNN(j)); % Andrew Ng
        delta_out = (outNN(j) - Y_train(j)) * dsigmoid(outNN(j)) ; % Andrew Ng
        
        % For Hidden layer (Observe that we are not calculating the bias)
        delta_a1 = (weights(3,2)*delta_out) * dsigmoid(Z_a1);
        delta_a2 = (weights(3,3)*delta_out) * dsigmoid(Z_a2);        
        
        % Accumulate deltas (Used by batch Gradient descent)
        for k = 1:num_layers
            if k == 1 % Bias cases
                accDelta(1,k) = accDelta(1,k) + (bias*delta_a1);
                accDelta(2,k) = accDelta(2,k) + (bias*delta_a2);
                accDelta(3,k) = accDelta(3,k) + (bias*delta_out);
            else % When k=2 or 3 input cases to neurons
                % Update with learnRate * Activations * smallDelta
                accDelta(1,k) = accDelta(1,k) + (X(j,1)*delta_a1);
                accDelta(2,k) = accDelta(2,k) + (X(j,2)*delta_a2);
                accDelta(3,k) = accDelta(3,k) + (x2(k-1)*delta_out);
            end
        end
        
        %%% Stochastic Gradient descent
        %Update at every new input....
        %
        % Delta: CurrentLayerActivations * delta(nextLayer)
        for k = 1:num_layers
            if k == 1 % Bias cases
                %weights(1,k)=weights(1,k) - learnRate*(bias*dsigmoid(Z3)delta2_1);
                %weights(2,k)=weights(2,k) - learnRate*(bias*delta2_2);
                %weights(3,k)=weights(3,k) - learnRate*bias*(delta_out);
            else % When k=2 or 3 input cases to neurons
                % Update with learnRate * Activations * smallDelta
                %weights(1,k)=weights(1,k) - learnRate*(X(j,1)*delta2_1);
                %weights(2,k)=weights(2,k) - learnRate*(X(j,2)*delta2_2);
                %weights(3,k)=weights(3,k) - learnRate*(x2(k-1)*delta_out);
            end
        end
    end
    %%% Gradient descent
    % Update after passing on all elements of training
    % Delta: CurrentLayerActivations * delta(nextLayer)
    %D = accDelta / sizeTraining;
    D = accDelta;
    % Substitute by the approximate derivative (true but slow)
    %D = approxDerivative;
    for k = 1:num_layers
        if k == 1 % Bias cases
            weights(1,k) = weights(1,k) - learnRate*(D(1,k));
            weights(2,k) = weights(2,k) - learnRate*(D(2,k));
            weights(3,k) = weights(3,k) - learnRate*(D(3,k));
        else % When k=2 or 3 input cases to neurons
            % Update with learnRate * Activations * smallDelta
            weights(1,k) = weights(1,k) - learnRate*(D(1,k));
            weights(2,k) = weights(2,k) - learnRate*(D(2,k));
            weights(3,k) = weights(3,k) - learnRate*(D(3,k));
        end
    end
    
    %%% Cost function calculation
    % After all calculations on the epoch calculate the cost function
    % Calculate Cost function output
    % Calculate p (used on regression)
    % Not ready yet for non-vectorized regularization
    p = sum(sum(weights.^2, 2));
    J = lossFunction.getLoss(outNN,Y_train) + ...
        regularization*p/(2*sizeTraining);
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
        Z_a1 = bias*weights(1,1) + test(1)*weights(1,2)...
            + test(2)*weights(1,3);
        x2(1) = sigmoid(Z_a1);
        a1 = sigmoid(Z_a1);
        
        % Second Neuron hidden layer
        Z_a2 = bias*weights(2,1) + test(1)*weights(2,2) ...
            + test(2)*weights(2,3);
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
xlabel('Epochs');
ylabel('Cost');