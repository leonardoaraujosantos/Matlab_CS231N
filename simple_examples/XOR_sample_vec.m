%% Vectorized XOR example
% Implement a simple Neural network to handle the XOR problem with a 3
% layer perceptron (MLP) with 2 neurons on the input, 2 on the hidden layer
% and 1 on the output, we also use bias.
%
% <<../../docs/imgs/XOR_NeuralNetwork.txt.png>>
%
% Every neuron on this ANN is connected to 3 weights, 2 weights coming from
% other neurons connections plus 1 connection with the bias, you can
% consider this as a 9 parmeter function.
%
% Just to remember how to calculate the neuron output
%
% <<../../docs/imgs/neural-net.png>>
%

%% Problem
% The XOR problem that caused to AI winter simply asked a perceptron to
% separate the blue and red classes, just to remeber a perceptron will look
% for a line that separate the two classes (Try yourself to separate those
% classes with a line)
%
% <<../../docs/imgs/XorDecision.png>>
%

%% Define training dataset
%
% <<../../docs/imgs/XorTable.png>>
%
% XOR input for x1 and x2
X = [0 0; 0 1; 1 0; 1 1];
% Desired output of XOR
Y_train = [0;1;1;0];
sizeTraining = size(X,1);

%% Define the sigmoid and dsigmoid
% Define the sigmoid (logistic) function and it's first derivative
sigmoid = @(x) 1.0 ./ ( 1.0 + exp(-x) );
dsigmoid = @(x) sigmoid(x) .* ( 1 - sigmoid(x) );

%% Initialization of meta parameters
% Learning coefficient
learnRate = 2; % Start to oscilate with 15
regularization = 0.0005;
% Number of learning iterations
epochs = 1000;
smallStep = 0.0001;

%% Define neural network structure
% More neurons means more local minimas, which means easier training and
% even if you get stuck on a local minima it's error will be close to the
% global minima (For large deep networks)
%
% Who is afraid of non convex loss-functions?
%
% http://videolectures.net/eml07_lecun_wia/
%
% http://www.cs.nyu.edu/~yann/talks/lecun-20071207-nonconvex.pdf
%
input_layer_size = 2;
hidden_layer_size = 2;
output_layer_size = 1;

%% Cost function definition
% On this case we will use the Cross entropy cost(or loss) function, the
% idea of the loss function is to give a number to show how bad/good your
% current set of parameters are. Here the definition of good means that our
% ANN output matches the training dataset. There are a lot of loss/cost
% functions out there but it seems that people are tending to use now more
% Cross-Entropy
%
% http://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/
%
% $\mathcal{L}(X, Y) = -\frac{1}{n} \sum_{i=1}^n y^{(i)} \ln a(x^{(i)}) + \left(1 - y^{(i)}\right) \ln \left(1 - a(x^{(i)})\right)$
%
%
% <include>CrossEntropy.m</include>
%
J_vec = zeros(1,epochs);
lossFunction = CrossEntropy();

%% Weights random initialization
% Initialize all the weights(parameters) of the neural network, layer by
% layer, the rule to create them is W(layer)=[next_layer X prev_layer+1].
% Where W1 is the layer that map the activations of the input layer to the
% hidden layer and W2 map the activations of the hidden layer to the output
%
% The way that the random values are distributed changes a lot the training
% performance, check chapter 3 on coursera (Weight initialization)
%
% http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
%
%rand('state',0); % Put random number generator on state 0
% Same thing but new syntax
rng(0,'v5uniform');
INIT_EPISLON = 0.8;
W1 = rand(hidden_layer_size,input_layer_size+1) ...
    * (2*INIT_EPISLON) - INIT_EPISLON;
W2 = rand(output_layer_size,hidden_layer_size+1) ...
    * (2*INIT_EPISLON) - INIT_EPISLON;
% Override manually to debug both vectorized and non vectorized
% implementation
%W1 = [-0.7690    0.6881   -0.2164; -0.0963    0.2379   -0.1385];
%W2 = [-0.1433   -0.4840   -0.6903];

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
    % <<../../docs/imgs/GradientChecking.PNG>>
    %
    %
    % <<../../docs/imgs/GradientChecking2.png>>
    %
    %
    % http://www.coursera.org/learn/machine-learning/lecture/Y3s6r/gradient-checking
    %
    Thetas = [W1(:) ; W2(:)];
    numgrad = zeros(size(Thetas));
    perturb = zeros(size(Thetas));    
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
        p_loss1 = sum(sum(nW1(:, 2:end).^2))+ ...
            sum(sum(nW2(:, 2:end).^2));        
        
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
        p_loss2 = sum(sum(nW1(:, 2:end).^2))+...
            sum(sum(nW2(:, 2:end).^2));        
        
        % Forward prop
        A1 = [ones(sizeTraining, 1) X];
        Z2 = A1 * nW1';
        A2 = sigmoid(Z2);
        A2 = [ones(sizeTraining, 1) A2];
        Z3 = A2 * nW2';
        A3 = sigmoid(Z3);
        h2 = A3;
        
        % Calculate both losses...        
        loss1 = lossFunction.getLoss(h1,Y_train) + ...
            (regularization/(2*sizeTraining)).*p_loss1;
        loss2 = lossFunction.getLoss(h2,Y_train) + ...
            (regularization/(2*sizeTraining)).*p_loss2;        
        
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
    % weights on the neural network (on our case 9 weights).
    % Backpropagation, at its core, simply consists of repeatedly applying 
    % the chain rule through all of the possible paths in our network.     
    % However, there are an exponential number of directed paths from the 
    % input to the output. Backpropagation's real power arises in the form 
    % of a dynamic programming algorithm, where we reuse intermediate 
    % results to calculate the gradient.     
    % We transmit intermediate errors backwards through a network, 
    % thus leading to the name backpropagation. In fact, backpropagation is
    % closely related to forward propagation, but instead of propagating 
    % the inputs forward through the network, we propagate the 
    % error backwards.    
    %
    % <<../../docs/imgs/ChainRule_1.png>>
    %
    %
    % <<../../docs/imgs/BackpropagationAlgorithm.png>>            
    %
    %
    % <<../../docs/imgs/BackwardPropagation_Vectorized.png>>
    %
    
    %%% Forward pass
    % Move from left(input) layer to the right (output), observe that every
    % previous activation get a extra collumn of ones (Include Bias)    
    %
    % <<../../docs/imgs/forwardPropagation.png>>
    %
    % First Activation (Input-->Hidden)    
    % Add extra collumn to A1
    %%% Bias Trick
    % This idea of adding a collumn of ones (or an extra row of ones, 
    % depending on how you created your weights) is called Bias trick, the
    % idea is that you can accelerate computation by doing just one 
    % multiplication and represent the weight and bias on a single matrix W
    %
    % <<../../docs/imgs/BiasTrick.jpeg>>
    %
    A1 = [ones(sizeTraining, 1) X]; 
    Z2 = A1 * W1';
    A2 = sigmoid(Z2);
    
    % Second Activation (Hidden-->Output)
    A2=[ones(sizeTraining, 1) A2];
    Z3 = A2 * W2';
    A3 = sigmoid(Z3);
    h = A3;
    
    %%% Backward pass
    % Move from right(output) layer to the left (input), actually stopping
    % on the last hidden layer before the input. Here we want to calculate
    % the error of every layer (desired - actual). The trap here is that
    % you calculate the output layer differenly than the input layers
    %
    % <<../../docs/imgs/BackwardPropagation.png>>
    %
    %
    % Output layer: (Why different tutorials have differ here?)  
    % Basically here is the derivative of the cost function related to the
    % neural network output, so this part here depends a lot on the cost
    % function that you choose    
    delta_output = (A3-Y_train); % Andrew Ng (Invert weight update signal)
    
    % Hidden layer, same idea of adding collumn of ones to include bias
    Z2=[ones(sizeTraining,1) Z2];    
    % Observe that we use the delta of the next layer
    % By the way dsigmoid(Z2) == A2 .* (1 - A2) (Could be slightly faster)
    delta_hidden=(delta_output*W2).*dsigmoid(Z2);    
    %delta_hidden=delta_output*W2.*(A2 .* (1 - A2));
    
    % Take out first column (bias column), to force the complete delta
    % to have the same size of it's respective weight
    delta_hidden=delta_hidden(:,2:end);
    
    % Calculate complete delta for every weight
    complete_delta_1 = (delta_hidden'*A1);
    complete_delta_2 = (delta_output'*A2);
    
    % Computing the partial derivatives with regularization, here we're
    % avoiding regularizing the bias term by substituting the first col of
    % weights with zeros and taking the first collumn of W
    p1 = ((regularization/sizeTraining)* ...
        [zeros(size(W1, 1), 1) W1(:, 2:end)]);
    p2 = ((regularization/sizeTraining)* ...
        [zeros(size(W2, 1), 1) W2(:, 2:end)]);
    D1 = (complete_delta_1 ./ sizeTraining) + p1;
    D2 = (complete_delta_2 ./ sizeTraining) + p2;
    
    %%% Check backpropagation
    % Compare backpropagation partial derivatives with numerical gradients
    % We should do this check just few times. This is because calculating
    % the numeric gradient every time is heavy.
    errorBackPropD1 = sum(sum(abs(D1 - numDeltaW1)));
    errorBackPropD2 = sum(sum(abs(D2 - numDeltaW2)));
    % Stop if backpropagation error bigger than 0.0001
    if (errorBackPropD1 > 0.0001) || (errorBackPropD2 > 0.0001)
        fprintf('Backpropagation error %d %d\n',...
            errorBackPropD1,errorBackPropD2);
        pause;
    end
    
    %%% Weight Update
    % Now we need to change the current set of weights with the negative
    % of the directions found on the back-propagation multiplied by some
    % factor (learning-rate). If we update the weight after backpropagating
    % the whole training set, this is called batch gradient descent. But if
    % we update the weight after backpropagating each sample on the
    % training set this is called Stochastic Gradient descent, which tends
    % to converge faster, but oscilate a little bit near the converging
    % point.
    %
    % http://www.holehouse.org/mlclass/17_Large_Scale_Machine_Learning.html
    %    
    %
    % <<../../docs/imgs/GradientDescent_Alone.png>>    
    %
    % <<../../docs/imgs/GradientDescent_And_StochasticGD.png>>
    %
    % The choice of the learning rate also affects the training performance
    %
    % <<../../docs/imgs/ChoiceLearningRate.jpg>>    
    %
    
    % Increment or decrement depending on delta_output sign
    W1 = W1 - learnRate*(D1);
    W2 = W2 - learnRate*(D2);
    
    %%% Calculate cost
    % After all calculations on the epoch are finished, calculate the cost
    % function between all your predictions (during) training and your
    % actual desired training output. The cost is a number representing how
    % well you are going on the training. For improving overfitting we also
    % include a regularization term that will avoid your training weights
    % get bigger values during training
    %
    % Calculate p (Regularization term)
    p = sum(sum(W1(:, 2:end).^2))+sum(sum(W2(:, 2:end).^2));
    % calculate Loss(or cost)    
    J = lossFunction.getLoss(h,Y_train) + ...
        (regularization/(2*sizeTraining)).*p;
    
    
    %%% Early stop
    % Stop if error is smaller than 0.01
    J_vec(i) = J;
    % Break if error is already low
    if J < 0.01
        J_vec = J_vec(1:i);
        break;
    end
end

%% Plot some information
% Plot Prediction surface and Cost vs epoch curve
testInpx1 = [-0.5:0.1:1.5];
testInpx2 = [-0.5:0.1:1.5];
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
        
        % Print XOR table
        if isequal(test,[0 0]) || ...
                isequal(test,[0 1]) || ...
                isequal(test,[1 0]) || ...
                isequal(test,[1 1])
           fprintf('%d XOR %d ==> %d\n',test(1),test(2),round(A3)); 
        end
    end
end
figure(2);
hold on;
plane = surf(X1, X2, testOut);
alpha(plane,.5);
plot3(0,0,0, '-o', 'MarkerSize',12, 'MarkerFaceColor','red');
plot3(0,1,1, '-o', 'MarkerSize',12, 'MarkerFaceColor','blue');
plot3(1,0,1, '-o', 'MarkerSize',12, 'MarkerFaceColor','blue');
plot3(1,1,0, '-o', 'MarkerSize',12, 'MarkerFaceColor','red');
title('Prediction surface');
view(-29, 52);
hold off;

figure(1);
plot(J_vec);
title('Cost vs epochs');

%% Some tutorials
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
% http://briandolhansky.com/blog/2013/9/27/artificial-neural-networks-backpropagation-part-4
% 
% http://playground.tensorflow.org/
% 
% http://caffe.berkeleyvision.org/tutorial/forward_backward.html
%
% http://caffe.berkeleyvision.org/tutorial/solver.html
%
% http://courses.cs.tau.ac.il/Caffe_workshop/Bootcamp/pdf_lectures/Lecture%203%20CNN%20-%20backpropagation.pdf
%
% http://www.vlfeat.org/matconvnet/matconvnet-manual.pdf
%
% http://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
%
% http://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
%
% http://learning.cs.toronto.edu/wp-content/uploads/2015/02/torch_tutorial.pdf
%
% http://hunch.net/~nyoml/torch7.pdf
%