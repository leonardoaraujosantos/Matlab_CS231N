classdef OutputSoftMaxLayer < BaseLayer
    %FULLYCONNECTEDLAYER Define the output softmax layer        
    properties
        typeLayer        
    end
    
    properties (Access = 'private')        
        numClasses
        lossFunction
    end
        
    methods (Access = 'public')
        function obj = OutputSoftMaxLayer(pNumClasses)
            % Initialize type
            obj.typeLayer = LayerType.OutputSoftMax;
            obj.numClasses = pNumClasses;     
            obj.lossFunction = SoftMaxLoss();
        end
        
        function [result] = forward(obj)
            result = [];
        end        
        
        function [gradient] = backward(obj)
            gradient = [];
        end
        
        % This will return the scores
        function [result] = getData(obj)
            result = [];
        end                
        
        function [type] = getType(obj)
            type = obj.typeLayer;
        end
        
        function [loss] = getLossFunction(obj)
            loss = obj.lossFunction;
        end
    end    
end

