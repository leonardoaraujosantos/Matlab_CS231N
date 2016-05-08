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
            obj.configs.learning_rate = 1e-3;
            obj.configs.momentum = 0.9;            
        end
        
        function [next_w] = sgd(obj, w, dw)
            
        end
        
        function [next_w] = sgd_momentum(obj, w, dw)           
            if ~isfield(obj.configs, 'velocity')
                velocity = zeros(size(w));
            else
                velocity = obj.configs.velocity;
            end
            
            next_w = w;
            
            learnRate = obj.configs.learning_rate;
            momentum = obj.configs.momentum;
            % Momentum
            velocity = (velocity * momentum) - (learnRate * dw);
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

