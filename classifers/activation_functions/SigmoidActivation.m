classdef SigmoidActivation < BaseActivationFunction
    %RELUACTIVATION SigmoidActivation Activation function (Old now)  
    % More informatio(python) here: 
    % https://databoys.github.io/Feedforward/
    % https://github.com/batra-mlp-lab/divmbest/search?utf8=%E2%9C%93&q=vl_dsigmoid
    % https://github.com/batra-mlp-lab/divmbest/blob/575b29142eb87bc690dc9629ee35d7e35f3a1f9e/pascalseg/external_src/cpmc_release1/external_code/vlfeats/toolbox/special/vl_sigmoid.m
    methods(Static)
        function [result] = forward_prop(x)
            result = (1./(1+exp(-x)));
        end
        
        function [result] = back_prop(x)
            t = single(1./(1+exp(-x)));
            result  = (t .* (1 - t));
        end
    end
    
end

