classdef LayerType < uint32
    % Define supported classification model layers on the system    
    enumeration
      Input (0)
      OutputSoftMax  (1)
      OutputSVM  (2)
      FullyConnected (3)
      OutputRegression (4)
      Pooling (5)
      Convolutional (6)
   end
    
end
