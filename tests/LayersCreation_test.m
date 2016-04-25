%% Neural network layers creation
% This project will use the layer creation scheme from convnetjs.
% URLS:
%
% http://cs.stanford.edu/people/karpathy/convnetjs/
%
% http://cs.stanford.edu/people/karpathy/convnetjs/
%
% Some examples (Javascript)
% 
% <<../../docs/imgs/sample_convjs_DeepNeuralNetwork_3HiddenLayer.PNG>>
%
% Convnet:
% 
% <<../../docs/imgs/sample_convjs_CNN_LeNet.PNG>>
%

%% Test 1: Simple Perceptron layers creation
% Perceptron
layers = LayerContainer;
layers <= struct('type',LayerType.Input,'rows',1,'cols',3,'depth',1);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',10,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.Output,'numClasses',10,'ActivationType',ActivationType.Sigmoid);
layers.showStructure();

%% Test 2: MLP layers creation
% Multi-layer-perceptron
layers = LayerContainer;
layers <= struct('type',LayerType.Input,'rows',1,'cols',3,'depth',1);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',10,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',10,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.Output,'numClasses',10,'ActivationType',ActivationType.Relu);
layers.showStructure();

%% Test 3: Autoencoder layers creation
% Autoencoder
layers = LayerContainer;
layers <= struct('type',LayerType.Input,'rows',28,'cols',28,'depth',1);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',50,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',25,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',1,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',25,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',50,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.OutputRegression,'numNeurons',28*28);
layers.showStructure();

%% Test 3: Convolutional neural network layers creation
% Convnet
layers = LayerContainer;
layers <= struct('type',LayerType.Input,'rows',32,'cols',32,'depth',3);
layers <= struct('type',LayerType.Convolutional,'kSize',5,'numFilters',64,'stride',1,'pad',2,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.Pooling,'kSize',2,'stride',2,'poolType',PoolingType.MaxPooling);
layers <= struct('type',LayerType.Convolutional,'kSize',3,'numFilters',128,'stride',1,'pad',1,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.Pooling,'kSize',2,'stride',2,'poolType',PoolingType.MaxPooling);
layers <= struct('type',LayerType.Convolutional,'kSize',3,'numFilters',128,'stride',1,'pad',1,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.Pooling,'kSize',2,'stride',2,'poolType',PoolingType.MaxPooling);
layers <= struct('type',LayerType.FullyConnected,'numNeurons',1024,'ActivationType',ActivationType.Relu);
layers <= struct('type',LayerType.Output,'numClasses',10,'ActivationType',ActivationType.Sigmoid);
layers.showStructure();