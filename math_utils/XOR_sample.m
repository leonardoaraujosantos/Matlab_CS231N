% Sigmoid and dSigmoid functions
sigmoid = @(x) 1.0 ./ ( 1.0 + exp(-x) );
dsigmoid = @(x) sigmoid(x) .* ( 1 - sigmoid(x) );

INIT_EPISLON = 0.1;

% XOR input for x1 and x2
input = [0 0; 0 1; 1 0; 1 1];
% Desired output of XOR
output = [0;1;1;0];
% Initialize the bias
bias = [1 1 1];
% Learning coefficient
learnRate = 0.5;
% Number of learning iterations
epochs = 10000;
% Calculate weights randomly using seed.
rand('state',sum(100*clock));
%weights = -1 +2.*rand(3,3);
weights = rand(3,3) * (2*INIT_EPISLON) - INIT_EPISLON;

for i = 1:epochs
   out = zeros(4,1);
   sizeTraining = length (input(:,1));
   for j = 1:sizeTraining
      % Hidden layer
      Z1 = bias(1,1)*weights(1,1) + input(j,1)*weights(1,2) + input(j,2)*weights(1,3);      
      x2(1) = sigmoid(Z1);
      a1 = sigmoid(Z1);
      
      Z2 = bias(1,2)*weights(2,1) + input(j,1)*weights(2,2)+ input(j,2)*weights(2,3);
      x2(2) = sigmoid(Z2);
      a2 = sigmoid(Z1);

      % Output layer
      Z3 = bias(1,3)*weights(3,1) + x2(1)*weights(3,2) + x2(2)*weights(3,3);
      out(j) = sigmoid(Z3);
      
      % Adjust delta values of weights
      % For output layer:
      % delta(wi) = xi*delta,
      % delta = (1-actual output)*(desired output - actual output) 
      %delta_out_layer = out(j)*(1-out(j))*(output(j)-out(j));
      %delta_out_layer = (1-out(j))*(output(j)-out(j));
      delta_out_layer = (output(j)-out(j));
      
      % Propagate the delta backwards into hidden layers
      delta2_1 = x2(1)*(1-x2(1))*weights(3,2)*delta_out_layer;
      delta2_2 = x2(2)*(1-x2(2))*weights(3,3)*delta_out_layer;
      
      % Add weight changes to original weights 
      % And use the new weights to repeat process.
      % delta weight = learnRate*x*delta
      for k = 1:3
         if k == 1 % Bias cases
            weights(1,k) = weights(1,k) + learnRate*bias(1,1)*delta2_1;
            weights(2,k) = weights(2,k) + learnRate*bias(1,2)*delta2_2;
            weights(3,k) = weights(3,k) + learnRate*bias(1,3)*delta_out_layer;
         else % When k=2 or 3 input cases to neurons
            weights(1,k) = weights(1,k) + learnRate*input(j,1)*delta2_1;
            weights(2,k) = weights(2,k) + learnRate*input(j,2)*delta2_2;
            weights(3,k) = weights(3,k) + learnRate*x2(k-1)*delta_out_layer;
         end
      end
   end   
end

fprintf('Outputs\n');
disp(round(out));

fprintf('\nWeights\n');
disp(weights);