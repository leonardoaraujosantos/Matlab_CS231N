classdef OutputSVMLayer < BaseLayer
    %FULLYCONNECTEDLAYER Define the output softmax layer        
    properties
        typeLayer        
    end
    
    properties (Access = 'private')        
        numClasses
        lossFunction
    end
        
    methods (Access = 'public')
        function obj = OutputSVMLayer(pNumClasses, svmDelta)
            % Initialize type
            obj.typeLayer = LayerType.OutputSVM;
            obj.numClasses = pNumClasses;     
            obj.lossFunction = SVMLoss(svmDelta);
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

