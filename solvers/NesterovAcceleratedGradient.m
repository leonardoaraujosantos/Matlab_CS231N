classdef NesterovAcceleratedGradient < BaseSolver
    % Implementation of NAG 
        
    properties
        typeSolver
        base_lr    % Learning rate used when training starts
        gamma      % Drop the learning rate at every step size
        stepsize   % Defines the number of iterations before update the lr
        epochs     % Number of times that we will iterate over the training
        batch_size % Number of samples from the training set on mini-batch
    end
    
    methods
        function obj = NesterovAcceleratedGradient(learningRate, batchSize,epochs)
            obj.typeSolver = SolverType.NesterovAcceleratedGradient;
            obj.base_lr = learningRate;
            obj.batch_size = batchSize; % Minibatch of batchSize > 1 
            obj.epochs = epochs;
        end
        
        function [weights, timeElapsed] = optimize(obj,model)
            tic;
            timeElapsed = toc;
        end
        
        function [type] = getType(obj)
           type = obj.typeSolver; 
        end
    end    
end

