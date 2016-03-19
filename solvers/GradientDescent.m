classdef GradientDescent < BaseSolver
    % Implementation of simple Gradient descent (batch)    
    % https://en.wikipedia.org/wiki/Gradient_descent
    %
    % The Gradient descent will pass on all training samples before update
    % the weights
    properties
        typeSolver
        base_lr    % Learning rate used when training starts
        gamma      % Drop the learning rate at every step size
        stepsize   % Defines the number of iterations before update the lr
        epochs     % Number of times that we will iterate over the training
        batch_size % Number of samples from the training set on mini-batch
    end
    
    methods
        function obj = GradientDescent(learningRate,epochs)
            obj.typeSolver = SolverType.GradientDescent;
            obj.base_lr = learningRate;
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

