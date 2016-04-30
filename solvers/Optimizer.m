classdef Optimizer < handle
    %OPTIMIZER Weight optimization class
    % This file implements various first-order update rules that are 
    % commonly used for training neural networks. Each update rule accepts 
    % current weights and the gradient of the loss with respect to those 
    % weights and produces the next set of weights. Each update rule 
    % has the same interface
    
    properties
        configs
    end
    
    methods
        function obj = Optimizer()
            obj.configs.learnRate = 1e-2;
            obj.configs.momentum = 0.9;
        end
        
        function [next_w] = sgd(obj, w, dw)
            
        end
        
        function [w] = sgd_momentum(obj, w, dw)
            momentum = obj.configs.momentum;
            velocity = zeros(size(w));
            
            next_w = w;
            
            % Momentum
            next_w = next_w + velocity;
            
            % Save velocity
            obj.configs.velocity = velocity;
        end
        
        function [w] = rms_prop(obj, w, dw)
            
        end
        
        function [w] = adam(obj, w, dw)
            
        end
    end
    
end

