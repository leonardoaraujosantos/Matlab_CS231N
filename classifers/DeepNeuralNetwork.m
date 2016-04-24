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
        trainingLossFunction
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
            obj.feedForward();
            
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
                    % Calculate difference for output layer
                    smallDelta{idxLayer} = (curLayer.getActivations - target);
                else
                    % Calculate difference for hidden layer
                    smallDelta{idxLayer} = (smallDelta{idxLayer+1} * (curLayer.weights)) .* curLayer.backPropagate();
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
        
        function [scores] = feedForward(obj)
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
                    currLayer.weights = rand(nextLayer.getNumNeurons,currLayer.getNumNeurons+1) * (2*INIT_EPISLON) - INIT_EPISLON;
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
                    
                    % After every epoch calculate the error function
                    lastLayer = obj.layers.getLayer(obj.layers.getNumLayers);
                    h = lastLayer.activations;
                    J = obj.trainingLossFunction.getLoss(h,batchLabels);
                    obj.lossVector(iterCounter) = J;
                    iterCounter = iterCounter + 1;
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
            scores = obj.feedForward();
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

