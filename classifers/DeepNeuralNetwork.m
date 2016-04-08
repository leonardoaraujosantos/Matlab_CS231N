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
    end
    
    methods (Access = 'private')
        % Find the partial derivative of the cost function related to every
        % parameter on the network (Vectorized form)
        function [deltas] = backPropagate(obj, feature, target)
            %% Do the forward propagation of the DNN
            % Set the input layer to the new X vector (or features)
            firstLayer = obj.layers.getLayer(1);
            if firstLayer.getType == LayerType.Input
                firstLayer.setActivations(feature);
            end
            obj.feedForward();
            
            %% Backpropagation algorithm
            % Reverse iterate on the Neural network layers (Don't including
            % first input layer)
            smallDelta = cell(obj.layers.getNumLayers,1);
            for idxLayer=obj.layers.getNumLayers:-1:2
                curLayer = obj.layers.getLayer(idxLayer);
                
                % Calculate offset between expected output and the current
                % output
                if curLayer.getType == LayerType.Output
                    % Calculate difference for output layer
                    smallDelta{idxLayer} = curLayer.getActivations - target;
                else
                    % Calculate difference for hidden layer
                    smallDelta{idxLayer} = ((curLayer.weights)' * smallDelta{idxLayer+1}) .* curLayer.backPropagate()';
                end
            end
            % Calculate the complete Deltas TODO delta is not that simple
            deltas = cell(obj.layers.getNumLayers,1);
            for idxLayer=1:obj.layers.getNumLayers-1
                curLayer = obj.layers.getLayer(idxLayer);
                deltas{idxLayer} = smallDelta{idxLayer+1}' * [1 curLayer.activations]';
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
        function obj = DeepNeuralNetwork(layers, solver)
            obj.layers = layers;
            obj.solver = solver;
            
            % Initialize randomicaly all the weights
            % Symmetry breaking (Coursera Machine learning course)
            INIT_EPISLON = 1;
            for idx=1:layers.getNumLayers
                currLayer = layers.getLayer(idx);
                if (idx+1) <= layers.getNumLayers
                    nextLayer = layers.getLayer(idx+1);
                    currLayer.weights = rand(currLayer.getNumNeurons+1,nextLayer.getNumNeurons) * (2*INIT_EPISLON) - INIT_EPISLON;
                    % Weights are a column vector
                    currLayer.weights = currLayer.weights';
                end
            end
        end
        
        function [timeElapsed] = train(obj, X_vec, Y_vec)
            tic;
            % Shuffle the dataset
            ind = PartitionDataSet.getShuffledIndex(size(Y_vec,1));
            X_vec = X_vec(ind,:);
            Y_vec = Y_vec(ind,:);
            
            % If needed extract a mini-batch
            miniBatchSize = obj.solver.batch_size;
            epochs = obj.solver.epochs;
            initialIndex = 1;
            for idxEpoch=1:epochs
                % Extract a chunk(if possible) from the training
                if (initialIndex+miniBatchSize < size(X_vec,1))
                    batchFeatures = X_vec(initialIndex:initialIndex+miniBatchSize,:);
                    batchLabels = Y_vec(initialIndex:initialIndex+miniBatchSize,:);
                    initialIndex = initialIndex + miniBatchSize;
                else
                    batchFeatures = X_vec(initialIndex:end,:);
                    batchLabels = Y_vec(initialIndex:end,:);
                end
                sizeBatch = size(batchFeatures,1);
                
                % Iterate on the whole training
                for idxTrain=1:sizeBatch
                    % Select sample from dataset
                    sampleFeatures = batchFeatures(idxTrain,:);
                    sampleTarget = batchLabels(idxTrain,:);
                    
                    % Run the back propagation to get the partial
                    % derivatives of the cost function related to every
                    % parameter on the neural network
                    deltas = obj.backPropagate(sampleFeatures, sampleTarget);
                    
                    % Update the weights on the minima (hopefully global
                    % minima) direction
                    for idxLayer=1:obj.layers.getNumLayers-1
                        curLayer = obj.layers.getLayer(idxLayer); 
                        curLayer.weights = curLayer.weights - (obj.solver.base_lr * deltas{idxLayer})';                        
                    end
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
    end
end

