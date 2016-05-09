classdef ConvolutionalLayer < BaseLayer
    %FULLYCONNECTEDLAYER Define the convolutional layer
    %   Actually this is the normal hidden layer on Neural Networks
    % More information:
    % http://www.slideshare.net/deview/251-implementing-deep-learning-using-cu-dnn
    
    properties
        typeLayer
        weights
        activations
        previousInput
        biasWeights
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
            % N (input volume), F(output volume)
            % C channels
            % H (rows), W(cols)
            [H, W, C, N] = size(activations);
            [HH, WW, C, F] = size(theta);
            
            size_out_H = ((H + (2*obj.numPad) - HH) / obj.stepStride) + 1;
            size_out_W = ((W + (2*obj.numPad) - WW) / obj.stepStride) + 1;
            
            % Pad if needed
            if (obj.numPad > 0)
                activations = padarray(activations,[obj.numPad obj.numPad 0 0]);
            end
            
            result = zeros(size_out_H,size_out_W,N,F);
            
            for idxInputDepth=1:N
                for idxOutputDepth=1:F                    
                    weights = theta(:,:,:,idxOutputDepth);
                    input = activations(:,:,:,idxInputDepth);                    
                    resConv = convn_vanilla(input,weights,obj.stepStride);

                    result(:,:,idxInputDepth,idxOutputDepth) = resConv + biasWeights(idxOutputDepth);
                end
            end
            % Save the previous inputs for use later on backpropagation
            obj.previousInput = activations;
            obj.weights = theta;
            obj.biasWeights = biasWeights;
        end
        
        function [dx, dw, db] = backPropagate(obj, dout)
            dw = zeros(;
            dx = 0;
            db = 0;
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

