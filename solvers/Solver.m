classdef Solver < handle
    %SOLVER Encapsulate all the logic for training, like, separating stuff
    % on batches, shuffling the dataset, and updating the weights with a
    % policy defined on the class optimizer
    
    
    properties
        optimizer
        learn_rate_decay
        batchSize
        epochs
        model
        lossVector
        trainAccuracyVector
        validationAccuracyVector
        X_val
        Y_val
    end
    
    methods (Access = 'private')
        function reset(obj)
            
        end
        
        function step(obj)
            
        end
        
        function checkAccuracy(obj)
            
        end
    end
    
    methods
        function obj = Solver(model, optimizer)            
            obj.model = model;            
            obj.optimizer = optimizer;
        end
        
        function train(obj, X_vec, y_vec)
            
        end
        
        function setValidation(obj, X_val, Y_val)
            
        end
    end
    
end

