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
            
            % Calculate output size, and allocate result
            size_out_H = ((H + (2*obj.numPad) - HH) / obj.stepStride) + 1;
            size_out_W = ((W + (2*obj.numPad) - WW) / obj.stepStride) + 1;
            result = zeros(size_out_H,size_out_W,N,F);
            
            % Save the previous inputs for use later on backpropagation
            obj.previousInput = activations;
            obj.weights = theta;
            obj.biasWeights = biasWeights;
            
            % Pad if needed
            if (obj.numPad > 0)
                activations = padarray(activations,[obj.numPad obj.numPad 0 0]);
            end
            
            % Convolve for each input/output depth
            for idxInputDepth=1:N
                for idxOutputDepth=1:F
                    
                    % Select weights and inputs
                    weights = theta(:,:,:,idxOutputDepth);
                    input = activations(:,:,:,idxInputDepth);
                    
                    % Do naive(slow) convolution)
                    resConv = convn_vanilla(input,weights,obj.stepStride);
                    
                    % Add bias and store
                    result(:,:,idxInputDepth,idxOutputDepth) = resConv + biasWeights(idxOutputDepth);
                end
            end
        end
        
        function [dx, dw, db] = backPropagate(obj, dout)
            % N (input volume), F(output volume)
            % C channels
            % H (rows), W(cols)
            [H_R, W_R,F, N] = size(dout);
            [H, W,C, N] = size(obj.previousInput);
            [HH, WW,C, F] = size(obj.weights);
            S = obj.stepStride;
            % Pad if needed
            if (obj.numPad > 0)
                obj.previousInput = padarray(obj.previousInput,[obj.numPad obj.numPad 0 0]);
            end
            
            dx = zeros(size(obj.previousInput));
            dw = zeros(size(obj.weights));
            db = zeros(size(obj.biasWeights));
            
            % Calculate dx
            for n=1:N
                for depth=1:F
                    weights = obj.weights(:,:,:,depth);
                    for r=1:S:H
                        for c=1:S:W
                            input =dout(ceil(r/S),ceil(c/S),depth,n);
                            prod =  weights * input;
                            dx(r:(r+HH)-1,c:(c+WW)-1,:,n) = dx(r:(r+HH)-1,c:(c+WW)-1,:,n) + prod;
                        end
                    end
                end
            end
            
            % Delete padded rows
            dx = dx(1+obj.numPad:end-obj.numPad, 1+obj.numPad:end-obj.numPad,:,:);
            
            % Calculate dw            
            for n=1:N
                for depth=1:F                    
                    for r=1:H_R
                        for c=1:W_R
                            input =dout(r,c,depth,n);
                            weights = obj.previousInput(r*S:(r*S+HH)-1,c*S:(c*S+WW)-1,:,n);
                            prod =  weights * input;                            
                            dw(:,:,:,depth) = dw(:,:,:,depth) + prod;
                        end
                    end
                end
            end
            
            % Calculate db
            for depth=1:F
                selDoutDepth = dout(: , : , depth, :);
                db(depth) = sum( selDoutDepth(:) );
            end
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

