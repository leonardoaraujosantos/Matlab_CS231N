classdef Solver < handle
    %SOLVER Encapsulate all the logic for training, like, separating stuff
    % on batches, shuffling the dataset, and updating the weights with a
    % policy defined on the class optimizer
    % Example:
    % myModel = DeepLearningModel(layers);
    % optimizer = Optimizer();
    % optimizer.configs.learning_rate = 1e-3;
    % optimizer.configs.momentum = 0.9;
    % solver = Solver(myModel, optimizer);
    % solver.train();
    % myModel.loss(X);
    
    
    properties
        optimizer
        learn_rate_decay
        batchSize
        num_epochs
        epochs
        model
        lossVector
        trainAccuracyVector
        validationAccuracyVector
        X_val
        Y_val
        X_train
        Y_train
        best_val_acc
        bestParameters
        verbose
        print_every
    end
    
    methods (Access = 'private')
        function reset(obj)
            
        end
        
        function step(obj)
            % Extract a randomic mini batch from the training, so basically
            % we extract some pieces of the training data on a random order
            numTrainTotal = size(obj.X_train,4);
            batch_mask = PartitionDataSet.getRandomBatchIndex(numTrainTotal, obj.batchSize);
            X_batch = obj.X_train(:,:,:,batch_mask);
            Y_batch = obj.Y_train(batch_mask,:);
            
            % Compute the loss and gradient
            [~,grads,loss] = obj.model.loss(X_batch, Y_batch);
            obj.lossVector =[obj.lossVector; loss];
            
            % Perform a parameter update, for every
            % layer. Some layers have two seta of parameters
            idxOptimizerConf = 1;
            for idxLayerIdx = 1:length(grads)
                elementGrads = grads{idxLayerIdx};
                if ~isempty(elementGrads)
                    for idxPar = 1:length(elementGrads)
                        dw = elementGrads{idxPar};                        
                        if (idxPar == 1)
                            w = obj.model.layers.getLayer(idxLayerIdx).weights;
                        else
                            w = obj.model.layers.getLayer(idxLayerIdx).biasWeights;
                        end
                        
                        % Optimize a little
                        nextW =obj.optimizer{idxOptimizerConf}.sgd_momentum(w,dw);
                        idxOptimizerConf = idxOptimizerConf + 1;
                        
                        if (idxPar == 1)
                            obj.model.layers.getLayer(idxLayerIdx).weights = nextW;
                        else
                            obj.model.layers.getLayer(idxLayerIdx).biasWeights = nextW;
                        end
                    end
                end
            end                        
        end
        
        function [accuracy] = checkAccuracy(obj, X,Y, sizeCheck)
            accuracy = 0;
            numTrainTotal = size(X,1);
            if sizeCheck ~= -1
                batch_mask = PartitionDataSet.getRandomBatchIndex(numTrainTotal, sizeCheck);
                X_batch = X(batch_mask,:);
                Y_batch = Y(batch_mask,:);
                testSize = sizeCheck;
            else
                X_batch = X;
                Y_batch = Y;
                testSize = numTrainTotal;
            end
            [scores,~] = obj.model.loss(X_batch);
            [~, trainedResults] = max(Y_batch,[],2);
            [~, modelResults] = max(scores,[],2);
            error = 0;
            for idxCheck=1:numTrainTotal
                if trainedResults(idxCheck) ~= modelResults(idxCheck)
                    error = error + 1;
                end
            end
            errorPercentage = (error*100) / testSize;
            accuracy = 100 -  errorPercentage;
        end
    end
    
    methods
        function obj = Solver(model, optimizer, data)
            obj.model = model;
            obj.epochs = 0;
            obj.best_val_acc = 0;
            obj.print_every = 100;
            obj.verbose = false;
            obj.learn_rate_decay = 1;
            obj.trainAccuracyVector = 0;
            obj.validationAccuracyVector = 0;
            obj.X_train = data{1};
            obj.Y_train = data{2};
            obj.lossVector = 0;
            if length(data) == 4
                obj.X_val = data{3};
                obj.Y_val = data{4};
            else
                obj.X_val = [];
                obj.Y_val = [];
            end
            
            % Make a copy of the optimizer for every layer with weights,
            % this is done because each layer could have it's own momentum
            % velocities (Multiplied by two due to the bias)
            for idxPar = 1:(obj.model.getNumLayersWithWeight*2)
                obj.optimizer{idxPar} = Optimizer();
                obj.optimizer{idxPar}.configs = optimizer.configs;
            end
        end
        
        function train(obj)
            num_train = size(obj.X_train,1);
            iterations_per_epoch = max(num_train / obj.batchSize, 1);
            num_iterations = obj.num_epochs * iterations_per_epoch;
            
            for t=1:num_iterations
                obj.step();
                if (obj.verbose) && (mod(t,obj.print_every) == 0)
                    fprintf('(Iteration %d / %d) loss: %f\n',...
                        t,num_iterations,obj.lossVector(end));
                end
                
                % At the end of every epoch, increment the epoch counter
                % and decay the learning rate.
                epoch_end = mod(t,iterations_per_epoch) == 0;
                if epoch_end
                    obj.epochs = obj.epochs + 1;
                    for idxPar = 1:(obj.model.getNumLayersWithWeight*2)
                        obj.optimizer{idxPar}.configs.learning_rate = obj.optimizer{idxPar}.configs.learning_rate ...
                            * obj.learn_rate_decay;
                    end                    
                end
                
                % Check train and val accuracy on the first iteration,
                % the last iteration, and at the end of each epoch.
                isFirst = (t == 0);
                isLast = (t == num_iterations);
                if isFirst || isLast || epoch_end
                    train_acc = obj.checkAccuracy(obj.X_train, ...
                        obj.Y_train, -1);
                    val_acc = obj.checkAccuracy(obj.X_val, ...
                        obj.Y_val, -1);
                    % Append accuracies
                    obj.trainAccuracyVector(end+1) = train_acc;
                    obj.validationAccuracyVector(end+1) = val_acc;
                    if (obj.verbose)
                        fprintf('(Epoch %d / %d) train acc: %f; val_acc: %f\n', ...
                            obj.epochs, obj.num_epochs, train_acc, val_acc);
                    end
                    
                    % Keep track of the best model
                    if val_acc > obj.best_val_acc
                        obj.best_val_acc = val_acc;
                        obj.bestParameters = obj.model.getModelParameters();
                    end
                end
            end
            % At the end of training swap the best params into the model
            %obj.model.setModelParams(obj.bestParameters);            
        end
        
        function setValidation(obj)
            
        end
    end
    
end

