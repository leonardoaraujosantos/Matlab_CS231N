classdef SoftmaxActivation < BaseActivationFunction
    % Softmax activation function, (dont confuse with softmax loss
    % function) 
    methods(Static)
        function [result] = forward_prop(x)
            fixX = x - max(x);
            e = exp(fixX);
            result = e / sum(e);
        end
        
        function [result] = back_prop(x)
            % TODO
            result = 0;
        end
    end
    
end

