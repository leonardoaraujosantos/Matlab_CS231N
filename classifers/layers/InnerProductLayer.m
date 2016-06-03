classdef InnerProductLayer < BaseLayer
    %InnerProductLayer Define the fully connected layer    
    
    properties
        typeLayer 
        weights
        activations
        z
        numOutputs
        previousInput
        biasWeights
    end
    
    properties (Access = 'private')        
        numberNeurons
        activationObject
        activationType            
    end
    
    methods (Access = 'public')
        function obj = InnerProductLayer(numOutputs)
            % Initialize type
            obj.typeLayer = LayerType.InnerProduct;  
            obj.numOutputs = numOutputs;
        end
        
        % Get number of neurons
        function [numNeurons] = getNumNeurons(obj)
            numNeurons = obj.numberNeurons;
        end
        
        function [activations] = fp(obj,prevLayerActivations)
            activations = obj.feedForward(prevLayerActivations, obj.weights, obj.biasWeights);
        end
        
        function [result] = feedForward(obj, activations, theta, biasWeights)
            %% Python version
            %N = x.shape[0]
            %a1 = x.reshape(N, -1)
            %out = a1.dot(w) + b.T                        
            %% Matlab (on the beginning we match the results with python)
            % Get number of inputs (depth N)             
            lenSizeActivations = length(size(activations));
            if (lenSizeActivations < 3)
                N = size(activations,1);
            else
                N = size(activations,ndims(activations)); % Matlab array highest dimension
            end                
            
            % On the python code we don't care about D size, it implicitly
            % let the reshape calculate it in our case we're using the
            % reshape_row_major that does not calculate implicitly...
            D = floor(numel(activations) / N);

            % Matlab reshape order is not the same as numpy, so to make the
            % same you need to transpose the dimensions of the input, than
            % reshape on the reverse order then transpose...
            % TODO: This was done to make results compatible with cs231n
            % assigments but in the end of the day is just a dot product
            % between activations and theta, plus 1*biasWeights, I dont
            % think that the order will change the results ...
            % a1 = reshape(activations,[N,D]);
            % permuting all dimensions [4 3 2 1] 
            %a1 = reshape(permute(activations,[ndims(activations):-1:1]),D,N)';
            a1 = reshape_row_major(activations,[N,D]);             
            % Dont care on python numeric match
            %a1 = reshape(activations,[N,D]); 
            %a1 = reshape(activations,N,[]); 
            result = (a1*theta) + (repmat(biasWeights,size(a1,1),1));
            
            % Save the previous inputs for use later on backpropagation
            obj.previousInput = activations;
            obj.weights = theta;
            obj.biasWeights = biasWeights;
            obj.activations = result;
            
        end
        
        function [dx, dw, db] = backPropagate(obj, dout)
            %% Python version
            % dx = np.dot(dout,w.T).reshape(x.shape)
            % dw = x.reshape(x.shape[0], -1).T.dot(dout)
            % db = np.sum(dout, axis=0)            
            
            %% Matlab (on the beginning we match the results with python)            
            % Get number of inputs (depth N)             
            lenSizeActivations = length(size(obj.previousInput));
            if (lenSizeActivations < 3)
                N = size(obj.previousInput,1);
            else
                N = size(obj.previousInput,ndims(obj.previousInput)); % Matlab array highest dimension
            end
            
            % On the python code we don't care about D size, it implicitly
            % let the reshape calculate it in our case we're using the
            % reshape_row_major that does not calculate implicitly...
            D = floor(numel(obj.previousInput) / N);
            
            inputSize = size(obj.previousInput);
            
            %% Calculate dx (same shape of previous input)                      
            dx = dout * obj.weights';            
            % Python format
            dx = reshape_row_major(permute(dx,[ndims(dx):-1:1]),fliplr(inputSize));            
            %dx = reshape(dx,inputSize);
                                    
            %% Calculate dw (same shape of previous weight)
            % reshape(matrix,firstDim,[]) will reshape the matrix with the
            % first dimension as firstDim and the rest automatically
            % calculated            
            x_reshape = reshape_row_major(obj.previousInput,[N,D]);
            %x_reshape = reshape(obj.previousInput,N,[]);
            dw = x_reshape' * dout;
            
            %% Calculate db (same shape as previous bias)
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
            descText = sprintf('InnerProduct numOutputs=%d \n',obj.numOutputs);
        end
        
        % Get number of parameters
        function [numParameters] = getNumParameters(obj)
            numParameters = numel(obj.weights);
        end
    end    
end

