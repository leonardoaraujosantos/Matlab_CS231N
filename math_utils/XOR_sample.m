%% Nice XOR example on matlab based on this tutorial
% http://matlabgeeks.com/tips-tutorials/neural-networks-a-multilayer-perceptron-in-matlab/
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
learnRate = 0.1;
% Number of learning iterations
epochs = 5000;
% Calculate weights randomly using seed.
INIT_EPISLON = 0.1;
weights = rand(3,3) * (2*INIT_EPISLON) - INIT_EPISLON;

for i = 1:epochs
   outNN = zeros(4,1);
   sizeTraining = length (X(:,1));
   for j = 1:sizeTraining
      % First Activation (Input-->Hidden)
      Z1 = bias(1,1)*weights(1,1) + X(j,1)*weights(1,2) + X(j,2)*weights(1,3);      
      x2(1) = sigmoid(Z1);
      a1 = sigmoid(Z1);
      
      % Second Activation (Hidden-->Output)
      Z2 = bias(1,2)*weights(2,1) + X(j,1)*weights(2,2)+ X(j,2)*weights(2,3);
      x2(2) = sigmoid(Z2);
      a2 = sigmoid(Z2);

      % Third Activation (Output-->outNN)
      Z3 = bias(1,3)*weights(3,1) + a1*weights(3,2) + a2*weights(3,3);
      outNN(j) = sigmoid(Z3);
      
      % Adjust delta values of weights
      % For output layer:
      % delta(wi) = xi*delta,
      % delta = (1-actual output)*(desired output - actual output) 
      %delta_out_layer = outNN(j)*(1-outNN(j))*(Y_train(j)-outNN(j));
      %delta_out_layer = (1-outNN(j))*(Y_train(j)-outNN(j));
      delta_out_layer = (Y_train(j)-outNN(j));
      
      % Propagate the delta backwards into hidden layers     
      delta2_2 = (weights(3,3)*delta_out_layer) * dsigmoid(Z2);
      delta2_1 = (weights(3,2)*delta_out_layer) * dsigmoid(Z1); %????
      
      % Add weight changes to original weights 
      % And use the new weights to repeat process.
      % delta weight = learnRate*x*delta
      for k = 1:num_layers
         if k == 1 % Bias cases
            weights(1,k) = weights(1,k) + learnRate*bias(1,1)*delta2_1;
            weights(2,k) = weights(2,k) + learnRate*bias(1,2)*delta2_2;
            weights(3,k) = weights(3,k) + learnRate*bias(1,3)*delta_out_layer;
         else % When k=2 or 3 input cases to neurons
            weights(1,k) = weights(1,k) + learnRate*X(j,1)*delta2_1;
            weights(2,k) = weights(2,k) + learnRate*X(j,2)*delta2_2;
            weights(3,k) = weights(3,k) + learnRate*x2(k-1)*delta_out_layer;
         end
      end
   end   
end

fprintf('Outputs\n');
disp(round(outNN));

fprintf('\nWeights\n');
disp(weights);