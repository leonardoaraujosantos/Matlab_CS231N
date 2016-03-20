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
        % Find the gradient of every layer
        function [timeElapsed] = backPropagate(obj, featues, targets)
            timeElapsed = 0;
            %% Do the forward propagation of the DNN
            %obj.feedForward();
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
                    nextLayer.activations = nextLayer.getActivation.forward_prop(weights' * activations);
                end
            end
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
                    currLayer.weights = rand(currLayer.getNumNeurons,nextLayer.getNumNeurons + 1) * (2*INIT_EPISLON) - INIT_EPISLON;
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
                % Run the back propagation (Update weights)
                obj.backPropagate(batchFeatures, batchLabels);
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

