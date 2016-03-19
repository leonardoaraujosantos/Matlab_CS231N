classdef (Abstract) BaseSolver < handle
    % Base class for solvers (used on DNN training)
    % Some links:
    %
    % http://caffe.berkeleyvision.org/tutorial/solver.html
    % http://caffe.berkeleyvision.org/tutorial/
    % http://research.microsoft.com/pubs/192769/tricks-2012.pdf
    properties (Abstract)
        typeSolver
        base_lr    % Learning rate used when training starts
        gamma      % Drop the learning rate at every step size
        stepsize   % Defines the number of iterations before update the lr
        epochs     % Number of times that we will iterate over the training
        batch_size % Number of samples from the training set on mini-batch
    end
    
    methods(Abstract, Access = public)
        [weights, timeElapsed] = optimize(obj,model);
        [type] = getType(obj);        
    end
    
end

