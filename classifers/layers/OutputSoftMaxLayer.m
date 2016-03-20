classdef OutputSoftMaxLayer < BaseLayer
    %FULLYCONNECTEDLAYER Define the output softmax layer        
    properties
        typeLayer
        weights
        activations
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
        
        % Get number of neurons
        function [numNeurons] = getNumNeurons(obj)
            numNeurons = obj.numClasses;
        end
        
        function [result] = feedForward(obj, inputs)
            result = [];
        end        
        
        function [gradient] = backPropagate(obj, targets)
            gradient = [];
        end
        
        % This will return the scores
        function [result] = getActivations(obj)
            result = obj.activations;
        end                
        
        function [type] = getType(obj)
            type = obj.typeLayer;
        end
        
        function [loss] = getLossFunction(obj)
            loss = obj.lossFunction;
        end
        
        function [descText] = getDescription(obj)
            descText = sprintf('OUTPUT_SOFTMAX num_classes=%d\n',obj.numClasses);
        end
    end    
end

