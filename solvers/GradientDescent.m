classdef GradientDescent < BaseSolver
    % Implementation of simple Gradient descent (batch)    
    % https://en.wikipedia.org/wiki/Gradient_descent
    % https://www.cs.toronto.edu/~hinton/csc2515/notes/lec6tutorial.pdf
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
        momentum   % Avoid get stuck on a local minima
        weightsPrevious % Previous weight
    end
    
    methods
        function obj = GradientDescent(learningRate,epochs)
            obj.typeSolver = SolverType.GradientDescent;
            obj.base_lr = learningRate;
            obj.epochs = epochs;
            obj.weightsPrevious = 0;
            obj.momentum = 0;
        end
        
        function [weights] = optimize(obj,pastWeights, delta)           
           weights = (obj.momentum*obj.weightsPrevious) + (1-obj.momentum)*(pastWeights - (obj.base_lr * delta));
           obj.weightsPrevious = weights;           
        end
        
        function [type] = getType(obj)
           type = obj.typeSolver; 
        end
    end
end

