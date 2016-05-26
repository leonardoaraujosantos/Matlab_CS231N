classdef LayerType < uint32
    % Define supported classification model layers on the system    
    enumeration
      Input (0)
      Output  (1)            
      FullyConnected (2)
      OutputRegression (3)
      Pooling (4)
      Convolutional (5)
      Relu (6)
      Sigmoid (7)
      Tanh(8)      
      InnerProduct(9)
   end
    
end
