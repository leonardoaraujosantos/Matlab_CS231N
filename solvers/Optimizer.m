classdef Optimizer < handle
    %OPTIMIZER Weight optimization class
    % This file implements various first-order update rules that are 
    % commonly used for training neural networks. Each update rule accepts 
    % current weights and the gradient of the loss with respect to those 
    % weights and produces the next set of weights. Each update rule 
    % has the same interface
    
    properties
    end
    
    methods
        function [w] = sgd(w, dw, configs)
            
        end
        
        function [w] = sgd_momentum(w, dw, configs)
            
        end
        
        function [w] = rms_prop(w, dw, configs)
            
        end
        
        function [w] = adam(w, dw, configs)
            
        end
    end
    
end

