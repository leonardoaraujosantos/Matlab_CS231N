classdef OutputSVMLayer < BaseLayer
    %FULLYCONNECTEDLAYER Define the output softmax layer        
    properties
        typeLayer        
    end
    
    properties (Access = 'private')        
        numClasses
        svmDelta
        lossFunction
    end
        
    methods (Access = 'public')
        function obj = OutputSVMLayer(pNumClasses, svmDelta)
            % Initialize type
            obj.typeLayer = LayerType.OutputSVM;
            obj.numClasses = pNumClasses;     
            obj.lossFunction = SVMLoss(svmDelta);
            obj.svmDelta = svmDelta;
        end
               
        function [result] = feedForward(obj, inputs)
            result = [];
        end        
        
        function [gradient] = backPropagate(obj, targets)
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
        
        function [descText] = getDescription(obj)
            descText = sprintf('OUTPUT_SVM num_classes=%d SVM_delta=%d\n',obj.numClasses, obj.svmDelta);
        end
    end    
end

