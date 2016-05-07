classdef ConvolutionalLayer < BaseLayer
    %FULLYCONNECTEDLAYER Define the convolutional layer
    %   Actually this is the normal hidden layer on Neural Networks
    % More information:
    % http://www.slideshare.net/deview/251-implementing-deep-learning-using-cu-dnn
    
    properties
        typeLayer
        weights
        activations
    end
    
    properties (Access = 'private')
        activationObject
        kernelSize
        numFilters
        stepStride
        numPad
    end
    
    methods (Access = 'public')
        function obj = ConvolutionalLayer(pKernelSize, pFilters, pStride, pPad)
            % Initialize type
            obj.typeLayer = LayerType.Convolutional;
            obj.kernelSize = pKernelSize;
            obj.numFilters = pFilters;
            obj.stepStride = pStride;
            obj.numPad = pPad;
        end
        
        % http://cs231n.github.io/convolutional-networks/#conv
        function [result] = feedForward(obj, activations, theta, biasWeights)
            %size_H = ((activations_H + (2*obj.numPad)) / obj.stepStride) + 1;
            %size_W = ((activations_W + (2*obj.numPad)) / obj.stepStride) + 1;
            result = 0;
        end
        
        function [gradient] = backPropagate(obj, dout)
            gradient = 0;
        end
        
        function [result] = getActivations(obj)
            result = 0;
        end
        
        function [type] = getType(obj)
            type = obj.typeLayer;
        end
        
        function [numN] = getKernelSize(obj)
            numN = obj.kernelSize;
        end
        
        function [numN] = getNumberOfFilters(obj)
            numN = obj.numFilters;
        end
        
        function [numN] = getStride(obj)
            numN = obj.stepStride;
        end
        
        function [numN] = getPad(obj)
            numN = obj.numPad;
        end
        
        function [actFunc] = getActivation(obj)
            actFunc = obj.activationObject;
        end
        
        % Get number of neurons
        function [numNeurons] = getNumNeurons(obj)
            numNeurons = [];
        end
        
        function [descText] = getDescription(obj)
            [~, names] = enumeration('ActivationType');
            descText = sprintf('CONV ksize=%d num_filters=%d stride=%d num_pad=%d Activation=%s\n',...
                obj.kernelSize,obj.numFilters,obj.stepStride,...
                obj.numPad,names{obj.activationType});
        end
        
        function [outSize] = getOutputSize(obj, inputSize)
            spatialSize = ((inputSize-obj.kernelSize)/obj.stepStride)+1;
            outSize = [spatialSize spatialSize obj.numFilters];
        end
        
        % Get number of parameters
        function [numParameters] = getNumParameters(obj)
            numParameters = ...
                (obj.kernelSize*obj.kernelSize*obj.numFilters)+1;
        end
    end
end

