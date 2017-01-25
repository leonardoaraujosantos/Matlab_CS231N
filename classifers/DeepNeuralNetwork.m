classdef (Sealed) DeepNeuralNetwork < BaseClassifer
    % Implementation of Deep Neural Network
    % Links of interest:
    %
    % http://scs.ryerson.ca/~aharley/neural-networks/
    % https://www.coursera.org/learn/machine-learning/home/week/4  (5,6,7)
    % https://class.coursera.org/neuralnets-2012-001
    % https://www.youtube.com/channel/UCPk8m_r6fkUSYmvgCBwq-sw
    % https://en.wikipedia.org/wiki/Backpropagation
    %
    % Some concepts:
    %
    % Backpropagation: Algorithm for finding all of the partial derivatives
    % in an ANN. Is called "back" because it calculate the derivatives from
    % the output layer to the input layer.
    %   The algorithm operates on the fact that the derivatives on one
    %   layer contain partial solutions to the derivatives on the layer
    %   below.(Use chain rule)
    % Optimizers(GD,SGD,ADA,etc...): Will be used inside the
    % backpropagation to update the weights of the neurons based on the
    % gradient found on the backpropagation.
    properties
        layers
        solver
        lossVector
        currentLoss
        trainingLossFunction
        verboseTraining
        dropOut
    end
    
    methods (Access = 'private')
        % Find the partial derivative of the cost function related to every
        % parameter on the network (Vectorized form)
        function [deltas] = backPropagate(obj, feature, target, prevDelta)
            sizeTraining = length(feature(:,1));            
            %% Do the forward propagation of the DNN
            % Set the input layer to the new X vector (or features)
            firstLayer = obj.layers.getLayer(1);
            if firstLayer.getType == LayerType.Input
                firstLayer.setActivations(feature);
            end
            obj.feedForward(1);
            
            % Calculate loss            
            lastLayer = obj.layers.getLayer(obj.layers.getNumLayers);
            h = lastLayer.activations;
            [obj.currentLoss, dL_dout] = obj.trainingLossFunction.getLoss(h,target);            
            
            %% Now the reverse propagation
            % Reverse iterate on the Neural network layers (Don't including
            % first input layer)
            % https://page.mi.fu-berlin.de/rojas/neural/chapter/K7.pdf
            smallDelta = cell(obj.layers.getNumLayers,1);
            for idxLayer=obj.layers.getNumLayers:-1:2
                curLayer = obj.layers.getLayer(idxLayer);
                
                % Calculate offset between expected output and the current
                % output
                if curLayer.getType == LayerType.Output
                    % Start with gradient of loss w.r.t correct class probability
                    smallDelta{idxLayer} = dL_dout;
                else
                    % Calculate difference for hidden layer
                    smallDelta{idxLayer} = (smallDelta{idxLayer+1} * (curLayer.weights)) .* curLayer.backPropagate();
                    % Re-apply the mask used on the forward propagation
                    if obj.dropOut > 0
                        smallDelta{idxLayer} = smallDelta{idxLayer} .* [ones(sizeTraining, 1) curLayer.dropoutMask];
                    end
                    % Take the bias
                    smallDeltaNoBias = smallDelta{idxLayer};
                    smallDeltaNoBias = smallDeltaNoBias(:,2:end);
                    smallDelta{idxLayer} = smallDeltaNoBias;
                end
            end
            
            % Calculate the complete Deltas
            deltas = cell(obj.layers.getNumLayers-1,1);
            for idxLayer=1:obj.layers.getNumLayers-1
                curLayer = obj.layers.getLayer(idxLayer);
                %deltas{idxLayer} = prevDelta{idxLayer} + (smallDelta{idxLayer+1}' * [ones(sizeTraining, 1) curLayer.activations]);
                deltas{idxLayer} = (smallDelta{idxLayer+1}' * [ones(sizeTraining, 1) curLayer.activations]);
            end
        end
        
        function [scores] = feedForward(obj, isTraining)
            scores = 0;
            for idx=1:obj.layers.getNumLayers
                currLayer = obj.layers.getLayer(idx);
                % If we have the input layer th
                activations = currLayer.getActivations;
                weights = currLayer.weights;
                % Get next layer if available
                if (idx+1) <= obj.layers.getNumLayers
                    nextLayer = obj.layers.getLayer(idx+1);
                    
                    % Calculate activations (Vectorized)
                    nextLayer.activations = nextLayer.feedForward(activations,weights);
                    
                    % Implement dropout (If needed)
                    % http://cs231n.github.io/neural-networks-2/#reg
                    if (isTraining && obj.dropOut > 0 && nextLayer.typeLayer == LayerType.FullyConnected)
                       % If we're on training phase and dropout is asked...
                       sizeAct = size(nextLayer.activations);
                       maskNeuronsLayer = (rand(sizeAct) < obj.dropOut) ...
                           / obj.dropOut;
                       nextLayer.activations = nextLayer.activations .* maskNeuronsLayer;
                       % Save mask (Going to be used on backpropagation)
                       nextLayer.dropoutMask = maskNeuronsLayer;
                    end
                end
            end
            
            % Scores will be the activation of the last layer
            scores = obj.layers.getLayer(obj.layers.getNumLayers).activations;
        end
    end
    
    methods (Access = 'public')
        function obj = DeepNeuralNetwork(layers, solver, varargin)
            obj.layers = layers;
            obj.solver = solver;
            obj.verboseTraining = false;
            obj.dropOut = 0;
            if nargin > 2
                obj.trainingLossFunction = varargin{1};
            end
            
            % Initialize randomicaly all the weights
            % Symmetry breaking (Coursera Machine learning course)
            INIT_EPISLON = 0.8;
            for idx=1:layers.getNumLayers
                currLayer = layers.getLayer(idx);
                if (idx+1) <= layers.getNumLayers
                    nextLayer = layers.getLayer(idx+1);
                    % Andrew Ng Machine learning course initialization
                    currLayer.weights = rand(nextLayer.getNumNeurons,currLayer.getNumNeurons+1) * (2*INIT_EPISLON) - INIT_EPISLON;
                    
                    % Cs231n Xavier Initialization
                    % http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
                    % http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
                    % http://uk.mathworks.com/help/stats/unifrnd.html?refresh=true
                    % http://uk.mathworks.com/help/matlab/ref/rand.html
                    %low = -sqrt(6.0/(currLayer.getNumNeurons+1 + nextLayer.getNumNeurons));
                    %high = sqrt(6.0/(currLayer.getNumNeurons+1 + nextLayer.getNumNeurons));
                    %currLayer.weights = low + (2*high) * rand(nextLayer.getNumNeurons,currLayer.getNumNeurons+1);
                    %currLayer.weights = rand(nextLayer.getNumNeurons,currLayer.getNumNeurons+1) / sqrt(currLayer.getNumNeurons+1);
                end
            end
        end                
        
        function [timeElapsed] = train(obj, X_vec, Y_vec)
            tic;
            
            % Initialize deltas, the deltas matrix has the same size/format
            %  of the weight matrix of the neural network
            deltas = cell(obj.layers.getNumLayers-1,1);
            for idxLayer=1:obj.layers.getNumLayers-1
                curLayer = obj.layers.getLayer(idxLayer);
                deltas{idxLayer} = zeros(size(curLayer.weights));
            end
            
            % Shuffle the dataset
            ind = PartitionDataSet.getShuffledIndex(size(Y_vec,1));
            X_vec = X_vec(ind,:);
            Y_vec = Y_vec(ind,:);
            
            % If needed extract a mini-batch
            miniBatchSize = obj.solver.batch_size;
            epochs = obj.solver.epochs;
            initialIndex = 1;
            
            % Initialize loss vector
            if isa(X_vec,'gpuArray')
                obj.lossVector = gpuArray(zeros(1,epochs));
            else
                obj.lossVector = zeros(1,epochs);
            end
            
            iterationsToCompleteTraining = size(X_vec,1)/miniBatchSize;
            iterCounter=1;
            
            for idxEpoch=1:epochs
                initialIndex = 1;                
                for idxIter=1:iterationsToCompleteTraining
                    % Extract a chunk(if possible) from the training
                    if (initialIndex+miniBatchSize < size(X_vec,1))
                        batchFeatures = X_vec(initialIndex:initialIndex+miniBatchSize-1,:);
                        batchLabels = Y_vec(initialIndex:initialIndex+miniBatchSize-1,:);
                        initialIndex = initialIndex + miniBatchSize;
                    else
                        % Get the rest
                        batchFeatures = X_vec(initialIndex:end,:);
                        batchLabels = Y_vec(initialIndex:end,:);
                    end
                    
                    % Vectorized backpropagation
                    deltas = obj.backPropagate(batchFeatures, batchLabels, deltas);
                    
                    % On Gradient Descent(Batch descent) the updates of the are
                    % weights is made after iterating on the whole training
                    % set, on Stochastic Gradient Descent (online training) we
                    % change the weights after every training example, on the
                    % mini-batch we update the weights after some training
                    % samples....
                    % Update the weights on the minima (hopefully global
                    % minima) direction
                    numItemsBatch = size(batchFeatures,1);                    
                    for idxLayer=1:obj.layers.getNumLayers-1
                        curLayer = obj.layers.getLayer(idxLayer);
                        curLayer.weights = obj.solver.optimize(curLayer.weights,deltas{idxLayer}./numItemsBatch);
                    end
                    
                    % Append calculated loss(During training FP) on loss vector                    
                    obj.lossVector(iterCounter) = obj.currentLoss;                    
                    iterCounter = iterCounter + 1;                    
                end
                if (obj.verboseTraining)
                    fprintf('Epoch %d/%d loss: %d\n',idxEpoch,epochs,obj.currentLoss); 
                end
            end
            timeElapsed = toc;
        end
        
        function [maxscore, scores, timeElapsed] = predict(obj, X_vec)
            tic;
            % Set X_vec on input layer
            firstLayer = obj.layers.getLayer(1);
            if firstLayer.getType == LayerType.Input
                firstLayer.setActivations(X_vec);
            end
            scores = obj.feedForward(0);
            [~, maxscore] = max(scores);
            timeElapsed = toc;
        end
        
        % Get number of parameters
        function [numParameters] = getNumParameters(obj)
            numParameters = 0;
            for idxLayer=1:obj.layers.getNumLayers-1
                curLayer = obj.layers.getLayer(idxLayer);
                numParameters = numParameters + ...
                    curLayer.getNumParameters();                
            end
        end
    end
end

