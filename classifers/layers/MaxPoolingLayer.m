classdef MaxPoolingLayer < BaseLayer
    %FULLYCONNECTEDLAYER Define the pooling layer
    % For more information refer here:
    % http://cs231n.github.io/convolutional-networks/
    % The pooling layer has no parameters and does not change the input
    % depth but it does change the previous activation (input) ratio (It's
    % width and height) imagine as a subsampling of the activation width
    % and height
    
    properties
        typeLayer
        weights
        activations
    end
    
    properties (Access = 'private')
        kernelSize
        stepStride
        poolingType
    end
    
    methods (Access = 'public')
        function obj = MaxPoolingLayer(pKernelSize, pStride)
            % Initialize type
            obj.typeLayer = LayerType.Pooling;
            obj.kernelSize = pKernelSize;
            obj.stepStride = pStride;
            obj.weights = 0;
            obj.poolingType = PoolingType.MaxPooling;                        
        end
        
        % Get number of neurons
        function [numNeurons] = getNumNeurons(obj)
            numNeurons = 0;
        end
        
        function [result] = feedForward(obj, activations)
            % N (input volume), F(output volume)
            % C channels
            % H (rows), W(cols)
            [H, W, C, N] = size(activations);            
            S = obj.stepStride;
            W_P = obj.kernelSize;
            H_P = obj.kernelSize;
            size_out_H = ((H - H_P) / S) + 1;
            size_out_W = ((W - W_P) / S) + 1;
            
            % Allocate output
            result = zeros(size_out_H,size_out_W,C,N);
            
            % Calculate Max-Pooling. Does like convolution mechanics but
            % instead of getting the window multiplied by heights, it just
            % return the biggest window value
            for n=1:N
                for depth=1:C     
                    input = activations(:,:,:,n);
                    resPool = max_pooling_vanilla(input,W_P,S);
                    result(:,:,:,n) = resPool;
%                     for r=1:S:H
%                         for c=1:S:W               
%                             % Get the sampling window
%                             window = activations((r*S)-1:(r*S+H_P)-1,(c*S)-1:(c*S+W_P)-1,depth,n);                            
%                             % Give the biggest value from the window
%                             result(r,c,depth,n) = max(window(:));
%                         end
%                     end
                end
            end
        end
        
        function [gradient] = backPropagate(obj, targets)
            gradient = 0;
        end
        
        function [result] = getActivations(obj)
            result = obj.activations;
        end
        
        function [type] = getType(obj)
            type = obj.typeLayer;
        end
        
        function [numN] = getKernelSize(obj)
            numN = obj.kernelSize;
        end
        
        function [numN] = getStride(obj)
            numN = obj.stepStride;
        end
        
        function [numN] = getPoolingType(obj)
            numN = obj.poolingType;
        end
        
        function [descText] = getDescription(obj)
            [~, names] = enumeration('PoolingType');            
            descText = sprintf('POOL ksize=%d stride=%d Type=%s\n',...
                obj.kernelSize,...
                obj.stepStride,names{obj.poolingType});
        end
        
        % Get number of parameters (No parameters on pooling layer)
        function [numParameters] = getNumParameters(obj)
            numParameters = 0;
        end
        
        function [outSize] = getOutputSize(obj, inputSize, prevDepth)
            spatialSize = ((inputSize-obj.kernelSize)/obj.stepStride)+1;
            % The pooling layer does not change the depth of the previous
            % layer, just scale it down (Sumarize information of the
            % previous layer)
            outSize = [spatialSize spatialSize prevDepth];
        end
    end
end

