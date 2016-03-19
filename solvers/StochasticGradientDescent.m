classdef StochasticGradientDescent < BaseSolver
    % Implementation of simple Stochastic Gradient Descent (online)    
    % https://en.wikipedia.org/wiki/Stochastic_gradient_descent
    % http://research.microsoft.com/pubs/192769/tricks-2012.pdf
    %
    % Use SGD when the size of the dataset is big
    % The SGD will update the weights at every sample on the dataset
    properties
        typeSolver
        base_lr    % Learning rate used when training starts
        gamma      % Drop the learning rate at every step size
        stepsize   % Defines the number of iterations before update the lr
        epochs     % Number of times that we will iterate over the training
        batch_size % Number of samples from the training set on mini-batch
    end
    
    methods
        function obj = StochasticGradientDescent(learningRate, batchSize,epochs)
            obj.typeSolver = SolverType.StochasticGradientDescent;
            obj.base_lr = learningRate;
            obj.batch_size = batchSize; % Minibatch of batchSize > 1 
            obj.epochs = epochs;
        end
        
        function [weights, timeElapsed] = optimize(obj,model)
           weights = 0;
           timeElapsed = 0;
        end
        
        function [type] = getType(obj)
           type = obj.typeSolver; 
        end
    end
end

