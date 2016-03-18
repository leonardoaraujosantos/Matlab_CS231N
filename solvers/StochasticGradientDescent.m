classdef StochasticGradientDescent < BaseSolver
    % Implementation of SGD
    properties
        typeSolver
        base_lr    % Learning rate used when training starts
        gamma      % Drop the learning rate at every step size
        stepsize   % Defines the number of iterations before update the lr
        epochs     % Number of times that we will iterate over the training
        batch_size % Number of samples from the training set on mini-batch
    end
    
    methods
    end
end

