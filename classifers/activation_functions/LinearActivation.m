classdef LinearActivation < BaseActivationFunction
    %Linear Activation (Can be used on the output layer)    
    methods(Static)
        function [result] = forward_prop(x)
            result = x;
        end
        
        function [result] = back_prop(x)            
            result  = 1;
        end
    end
    
end

