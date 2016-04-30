classdef Solver < handle
    %SOLVER Encapsulate all the logic for training, like, separating stuff
    % on batches, shuffling the dataset, and updating the weights with a
    % policy defined on the class optimizer
    
    
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
    end
    
    methods (Access = 'private')
        function reset(obj)
            
        end
        
        function step(obj)
            
        end
        
        function checkAccuracy(obj)
            
        end
    end
    
    methods
        function obj = Solver(model, optimizer)
            obj.model = model;
            obj.optimizer = optimizer;
            obj.epochs = 0;
            obj.best_val_acc = 0;
        end
        
        function train(obj)
            num_train = size(obj.X_vec);
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
                    obj.optimizer.learnRate = obj.optimizer.learnRate ...
                        * obj.learn_rate_decay;
                end
                
                % Check train and val accuracy on the first iteration,
                % the last iteration, and at the end of each epoch.
                isFirst = (t == 0);
                isLast = (t == num_iterations);
                if isFirst || isLast || epoch_end
                    train_acc = obj.checkAccuracy(obj.X_train, ...
                        obj.Y_train);
                    val_acc = obj.checkAccuracy(obj.X_val, ...
                        obj.Y_val);
                    % Append accuracies
                    obj.trainAccuracyVector(end+1) = train_acc;
                    obj.validationAccuracyVector(end+1) = val_acc;
                    if (obj.verbose)
                        fprintf('(Epoch %d / %d) train acc: %f; val_acc: %f', ...
                            obj.epoch, obj.num_epochs, train_acc, val_acc);
                    end
                    
                    % Keep track of the best model
                    if val_acc > obj.best_val_acc
                        obj.best_val_acc = val_acc;
                        obj.bestParameters = obj.model.params;
                    end
                end
            end
            % At the end of training swap the best params into the model
            obj.model.params = obj.bestParameters;
        end
        
        function setValidation(obj)
            
        end
    end
    
end

