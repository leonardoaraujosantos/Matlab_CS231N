classdef PartitionDataSet < handle
    %PARTITIONDATASET Divide your dataset on K trainning sets or in 2
    %(simple) partitioning
    % References:
    % https://en.wikipedia.org/wiki/Cross-validation_(statistics)
    % http://stackoverflow.com/questions/12630293/matlab-10-fold-cross-validation-without-using-existing-functions
    
    
    properties ( Access = 'private' )
        typeOfPartitioning
        original_train_X
        original_train_Y
        numTestSets
        numObservations
        testSize
        trainSize
        partitioned_train_X
        partitioned_train_Y
        partitioned_val_X
        partitioned_val_Y
        index_transform;
    end
    
    methods ( Access = 'private')
        function ShuffleData(obj)
            ind = randperm(obj.numObservations);
            obj.original_train_X = obj.original_train_X(ind,:);
            obj.original_train_Y = obj.original_train_Y(ind,:);
        end
    end
    
    methods(Static)
        % Static method used just to shuffle data
        function [shuffledIndex] = getShuffledIndex(numObservations)
            shuffledIndex = randperm(numObservations);
        end
    end
    
    methods ( Access = 'public' )
        function obj = PartitionDataSet(trainX, trainY)
            obj.original_train_X = trainX;
            obj.original_train_Y = trainY;
            obj.typeOfPartitioning = [];
            obj.numTestSets = 0;
            obj.numObservations = length(trainX);
            obj.partitioned_train_X = [];
            obj.partitioned_train_Y = [];
            obj.partitioned_val_X = [];
            obj.partitioned_val_Y = [];
        end
        
        % Do K partitioning
        % In k-fold partitioning, the original samples are divided
        % into k equal sized subsamples. Of the k subsamples,
        % a single subsample is retained as the validation data for testing
        % the model, and the remaining k − 1 subsamples are used as
        % training data.
        % The cross-validation process is then repeated k times
        % (the folds), with each of the k subsamples used exactly once as
        % the validation data. The k results from the folds can then be
        % averaged (or otherwise combined) to produce a single estimation.
        % The advantage of this method over repeated random sub-sampling
        % is that all observations are used for both training and
        % validation, and each observation is used for
        % validation exactly once.
        function doKPartitioning(obj, numK, doShuffle)
            obj.typeOfPartitioning = 1;
            obj.numTestSets = numK;
            if (doShuffle)
                obj.ShuffleData();
            end
            
            % Calculate train/validation sizes
            obj.testSize = ceil(obj.numObservations/numK);
            obj.trainSize = obj.numObservations - obj.testSize;
            
            % Calculate a transform that will be used to decide which
            % samples will populate each group
            obj.index_transform = 1 + mod((1:obj.numObservations)',numK);
        end
        
        % Do old partitioning just separating some part of the data for
        % trainning
        function doSimplePartitioning(obj, valPercentage, doShuffle)
            obj.typeOfPartitioning = 2;
            obj.numTestSets = 1;
            if (doShuffle)
                obj.ShuffleData();
            end
            obj.testSize = floor((valPercentage/100) * obj.numObservations);
            obj.trainSize = obj.numObservations - obj.testSize;
            obj.partitioned_train_X = obj.original_train_X(1:end-obj.testSize,:);
            obj.partitioned_train_Y = obj.original_train_Y(1:end-obj.testSize,:);
            obj.partitioned_val_X = obj.original_train_X(obj.numObservations-obj.testSize+1:end,:);
            obj.partitioned_val_Y = obj.original_train_Y(obj.numObservations-obj.testSize+1:end,:);
        end
        
        function num = getNumTestSets(obj)
            num = obj.numTestSets;
        end
        
        function num = getNumObservations(obj)
            num = obj.numObservations;
        end
        
        function num = getTestSize(obj)
            num = obj.testSize;
        end
        
        function num = getTrainSize(obj)
            num = obj.trainSize;
        end
        
        function [train_X, train_Y, val_X, val_Y] = getDataset(obj,numPart)
            train_X = [];
            train_Y = [];
            val_X = [];
            val_Y = [];
            if ~isempty( obj.typeOfPartitioning )
                if obj.typeOfPartitioning == 1
                    % K Partitioning
                    
                    % Calculate the indexes for a particular fold
                    train_index_mask = obj.index_transform ~= numPart;
                    val_index_mask = obj.index_transform == numPart;
                    
                    % Return the validation set
                    val_X = obj.original_train_X(val_index_mask,:);
                    val_Y = obj.original_train_Y(val_index_mask);
                    
                    % Return the trainning set
                    train_X = obj.original_train_X(train_index_mask,:);
                    train_Y = obj.original_train_Y(train_index_mask);
                    
                end
                if obj.typeOfPartitioning == 2
                    % Simple Partitioning only one partitioning available
                    if (numPart == 1)
                        train_X = obj.partitioned_train_X;
                        train_Y = obj.partitioned_train_Y;
                        val_X = obj.partitioned_val_X;
                        val_Y = obj.partitioned_val_Y;
                    end
                end
            end
        end
    end
end

