classdef InnerProductLayer < BaseLayer
    %InnerProductLayer Define the fully connected layer    
    
    properties
        typeLayer 
        weights
        activations
        z
        dropoutMask
        previousInput
        biasWeights
    end
    
    properties (Access = 'private')        
        numberNeurons
        activationObject
        activationType            
    end
    
    methods (Access = 'public')
        function obj = InnerProductLayer()
            % Initialize type
            obj.typeLayer = LayerType.FullyConnected;            
        end
        
        % Get number of neurons
        function [numNeurons] = getNumNeurons(obj)
            numNeurons = obj.numberNeurons;
        end
        
        function [result] = feedForward(obj, activations, theta, biasWeights)
            % Get number of inputs (depth N) 
            N = size(activations,ndims(activations));
            D = size(theta,1);

            % Matlab reshape order is not the same as numpy, so to make the
            % same you need to transpose the dimensions of the input, than
            % reshape on the reverse order then transpose...
            % TODO: This was done to make results compatible with cs231n
            % assigments but in the end of the day is just a dot product
            % between activations and theta, plus 1*biasWeights, I dont
            % think that the order will change the results ...
            % a1 = reshape(activations,size(activations,1),[])
            % permuting all dimensions [4 3 2 1] 
            a1 = reshape(permute(activations,[ndims(activations):-1:1]),[],size(activations,1))';
            result = (a1*theta) + (repmat(biasWeights,size(a1,1),1));
            
            % Save the previous inputs for use later on backpropagation
            obj.previousInput = activations;
            obj.weights = theta;
            obj.biasWeights = biasWeights;
            
        end
        
        function [dx, dw, db] = backPropagate(obj, dout)
            inputSize = size(obj.previousInput);
            dx = dout * obj.weights';
            %dx = reshape(dx,inputSize); TODO test like THIS on production
            
            % NOW TO PUT ON PYTHON FORMAT
            % Again to match python reshape we need to transpose the input,
            % reverse the reshape order and transpose again            
            % dx = reshape(dx,inputSize);
            dx = reshape(permute(dx,[ndims(dx):-1:1]),fliplr(inputSize));
            dx = permute(dx,[ndims(dx):-1:1]);
            
                                    
            % reshape(matrix,firstDim,[]) will reshape the matrix with the
            % first dimension as firstDim and the rest automatically
            % calculated
            x_reshape = reshape(permute(obj.previousInput,[ndims(obj.previousInput):-1:1]),[],inputSize(1))';
            dw = x_reshape' * dout;
            
            db = sum(dout,1);
        end
        
        function [result] = getActivations(obj)
            result = obj.activations;
        end
        
        function [type] = getType(obj)
            type = obj.typeLayer;
        end
        
        function [numN] = getNumberofNeurons(obj)
            numN = obj.numberNeurons;
        end
        
        function [actFunc] = getActivation(obj)
            actFunc = obj.activationObject;
        end
        
        function [descText] = getDescription(obj)
            [~, names] = enumeration('ActivationType');
            descText = sprintf('FC num_params=%d \n',numel(obj.weights) + numel(obj.biasWeights));
        end
        
        % Get number of parameters
        function [numParameters] = getNumParameters(obj)
            numParameters = numel(obj.weights);
        end
    end    
end

