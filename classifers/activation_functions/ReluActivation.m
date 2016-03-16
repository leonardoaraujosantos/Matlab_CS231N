classdef ReluActivation < BaseActivationFunction
    %RELUACTIVATION Relu Activation function (Standard for DNN)  
    % Some activation functions examples here:
    % http://www.mathworks.com/help/fuzzy/sigmf.html
    % http://scs.ryerson.ca/~aharley/neural-networks/#mjx-eqn-nodeFormula
    % http://www.mathworks.com/matlabcentral/fileexchange/38310-deep-learning-toolbox
    % http://ceit.aut.ac.ir/~keyvanrad/DeeBNet%20Toolbox.html
    % https://github.com/gopaczewski/coursera-ml
    methods(Static)
        function [result] = forward_prop(x)
            result = single(max(0,x));
        end
        
        function [result] = back_prop(x)
            result  = single(x > 0);
        end
    end
    
end

