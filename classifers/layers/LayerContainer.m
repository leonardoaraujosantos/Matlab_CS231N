classdef LayerContainer < BaseLayerContainer
    %LAYERCONTAINER Layer container implementation
    
    properties (Access = private)
        % It will be a cell there is no list on matlab
        layersList
        numLayers;
    end
    
    methods (Access = 'public')
        function obj = LayerContainer()
            obj.layersList = {};
            obj.numLayers = 0;
        end
        
        function pushLayer(obj,metaDataLayer)            
            switch metaDataLayer.type
                case LayerType.Input
                    % Rows, Cols, Depth
                    layerInst = InputLayer(metaDataLayer.rows,metaDataLayer.cols,metaDataLayer.depth);                
                case LayerType.Output
                    % Number of classes
                    layerInst = OutputLayer(metaDataLayer.numClasses,metaDataLayer.ActivationType);                
                case LayerType.FullyConnected
                    % Number of neurons, Activation type
                    layerInst = FullyConnectedLayer(metaDataLayer.numNeurons,metaDataLayer.ActivationType);
                case LayerType.InnerProduct
                    % Number of outputs
                    layerInst = InnerProductLayer(metaDataLayer.numOutputs);
                case LayerType.Relu                    
                    layerInst = ReluLayer();
                case LayerType.OutputRegression
                    % Number of neurons
                    layerInst = OutputRegressionLayer(metaDataLayer.numNeurons);
                case LayerType.Pooling
                    % Kernel size, Stride, PollType
                    layerInst = PoolingLayer(metaDataLayer.kSize,metaDataLayer.stride,metaDataLayer.poolType);
                case LayerType.Convolutional
                    % Kernel size, Number of Filters, Stride, Pad,
                    % Activation Type
                    layerInst = ConvolutionalLayer(metaDataLayer.kSize,metaDataLayer.numFilters,metaDataLayer.stride,metaDataLayer.pad,metaDataLayer.ActivationType);
            end
            obj.layersList{obj.numLayers+1} = layerInst;
            obj.numLayers = obj.numLayers + 1;
        end
        
        % Override the "<=" operator (used to push a new layer)
        function result = le(obj,B)            
            obj.pushLayer(B);
        end
        
        function removeLayer(obj,index)
            obj.numLayers = obj.numLayers - 1;
        end
        
        function layer = getLayer(obj,index)
            layer = obj.layersList{index};
        end
        
        function numLayers = getNumLayers(obj)
            numLayers = obj.numLayers;
        end
        
        function layerList = getLayersList(obj)
            layerList = obj.layersList;
        end
        
        function showStructure(obj)
            % Iterate on all layers
            for idxLayer=1:obj.numLayers
               layerInstance = obj.layersList{idxLayer};
               txtDesc = layerInstance.getDescription();
               fprintf('LAYER(%d)--> %s',idxLayer,txtDesc);
            end
        end
        
        function generateGraphVizStruct(obj)
            
        end
    end    
end

