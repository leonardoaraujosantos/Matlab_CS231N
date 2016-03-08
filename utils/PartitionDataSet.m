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
        partitioned_train_X
        partitioned_train_Y
        partitioned_val_X
        partitioned_val_Y
        groupData_K_X
        groupData_K_Y
    end
    
    methods ( Access = 'private')
        function ShuffleData(obj)
            ind = randperm(obj.numObservations);
            obj.original_train_X = obj.original_train_X(ind,:);
            obj.original_train_Y = obj.original_train_Y(ind,:);            
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
            obj.groupData_K_X = {};
            obj.groupData_K_Y = {};
        end
        
        % Do K partitioning
        % In k-fold cross-validation, the original sample is randomly 
        % partitioned into k equal sized subsamples. Of the k subsamples, 
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
            testSize = ceil(obj.numObservations/numK);
            trainSize = obj.numObservations - testSize;
            
            % Create 5 groups
            numItemsGroup = testSize;            
            stIndex = 1;
            for idxGroup=1:numK
                if (stIndex+numItemsGroup) < obj.numObservations
                    obj.groupData_K_X{idxGroup} = obj.original_train_X(stIndex:stIndex+numItemsGroup-1,:);
                    obj.groupData_K_Y{idxGroup} = obj.original_train_Y(stIndex:stIndex+numItemsGroup-1,:);
                else
                    obj.groupData_K_X{idxGroup} = obj.original_train_X(stIndex:end,:);
                    obj.groupData_K_Y{idxGroup} = obj.original_train_Y(stIndex:end,:);
                end
                stIndex = 1 + (idxGroup*numItemsGroup);
            end            
        end
        
        % Do old partitioning just separating some part of the data for
        % trainning
        function doSimplePartitioning(obj, valPercentage, doShuffle)
            obj.typeOfPartitioning = 2;
            obj.numTestSets = 1;
            if (doShuffle)
                obj.ShuffleData();
            end
            testSize = floor((valPercentage/100) * obj.numObservations);
            obj.partitioned_train_X = obj.original_train_X(1:end-testSize,:);
            obj.partitioned_train_Y = obj.original_train_Y(1:end-testSize,:);
            obj.partitioned_val_X = obj.original_train_X(obj.numObservations-testSize+1:end,:);
            obj.partitioned_val_Y = obj.original_train_Y(obj.numObservations-testSize+1:end,:);
        end
        
        function num = getNumTestSets(obj)
            num = obj.numTestSets;
        end
        
        function num = getNumObservations(obj)
            num = obj.numObservations;
        end
        
        function [train_X, train_Y, val_X, val_Y] = getDataset(obj,numPart)
            train_X = [];
            train_Y = [];
            val_X = [];
            val_Y = [];
            if ~isempty( obj.typeOfPartitioning )
                if obj.typeOfPartitioning == 1
                    % K Partitioning
                    K = obj.numTestSets;
                    % Now we have K groups (ex:5) and we want the dataset
                    % for the first fold, so on this case the validation
                    % set will be the first group and the trainning set all
                    % the others
                    groups = [1:1:K];
                    validation_index = numPart;
                    % Select all the others groups excluding the current
                    % fold
                    trainning_indexes = find(groups ~= validation_index);                    
                    
                    % Return the validation set
                    val_X = obj.groupData_K_X{validation_index};
                    val_Y = obj.groupData_K_Y{validation_index};
                    
                    % Return the trainning set
                    trainning_cell_groups_X = cell(length(trainning_indexes));
                    trainning_cell_groups_Y = cell(length(trainning_indexes));
                    for idxGroup=1:length(trainning_indexes)
                       group_idx = trainning_indexes(idxGroup);        
                       trainning_cell_groups_X{idxGroup} = obj.groupData_K_X{group_idx};
                       trainning_cell_groups_Y{idxGroup} = obj.groupData_K_Y{group_idx};
                    end
                    train_X = cell2mat(trainning_cell_groups_X);
                    train_Y = cell2mat(trainning_cell_groups_Y);
                    
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

