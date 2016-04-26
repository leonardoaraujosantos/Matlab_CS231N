classdef TanhActivation < BaseActivationFunction
    %RELUACTIVATION TanhActivation Activation function (Old now)  
    % More informatio(python) here: 
    % http://ufldl.stanford.edu/wiki/index.php/Neural_Networks
    % http://www.cs.bham.ac.uk/~jxb/INC/l7.pdf
    % https://theclevermachine.wordpress.com/2014/09/08/derivation-derivatives-for-common-neural-network-activation-functions/
    methods(Static)
        function [result] = forward_prop(x)
            result = (tanh(-x));
        end
        
        function [result] = back_prop(x)
            t = (tanh(-x));
            result  = (1 - (t.*t));
        end
    end
    
end

