import numpy as np

def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization (Use on FC layers).
  Use computation graph to guide the backward propagation!
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of: (dx(N,D), dgamma(D), dbeta(D))
  """
  dx, dgamma, dbeta = None, None, None

  # http://cthorey.github.io./backpropagation/
  mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
  eps = bn_param.get('eps', 1e-5)
  N, D = dout.shape

  # Backprop Step 9
  dva3 = dout
  dbeta = np.sum(dout, axis=0)
  # Backprop step 8
  dva2 = gamma * dva3
  dgamma = np.sum(va2 * dva3, axis=0)
  # Backprop step 7
  dxmu = invvar * dva2
  dinvvar = np.sum(xmu * dva2, axis=0)
  # Backprop step 6
  dsqrtvar = -1. / (sqrtvar**2) * dinvvar
  # Backprop step 5
  dvar = 0.5 * (var + eps)**(-0.5) * dsqrtvar
  # Backprop step 4
  dcarre = 1 / float(N) * np.ones((carre.shape)) * dvar
  # Backprop step 3
  dxmu += 2 * xmu * dcarre
  # Backprop step 2
  dx = dxmu
  dmu = - np.sum(dxmu, axis=0)
  # Basckprop step 1
  dx += 1 / float(N) * np.ones((dxmu.shape)) * dmu

  return dx, dgamma, dbeta

def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization (Use on FC layers).
  Input:
  - x: Data of shape (N, D)
  - gamma,beta: Scale/Shift parameter of shape (D,)
  - bn_param: Dictionary with the following keys: (mode,eps,momentum,r_mean/var)
  Returns a tuple of: (out, cache)
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    # Forward pass
    # Step 1: Calculate mean
    mu = 1 / float(N) * np.sum(x, axis=0)
    # Step 2: Subtract the mean of every training sample
    xmu = x - mu
    # Step 3 - Calculate denominator
    carre = xmu**2
    # Step 4 - Calculate variance
    var = 1 / float(N) * np.sum(carre, axis=0)
    # Step 5 - Add eps for numerical stability then get square root
    sqrtvar = np.sqrt(var + eps)
    # Step 6 - Invert square root
    invvar = 1. / sqrtvar
    # Step 7 - Calculate normalization
    va2 = xmu * invvar
    # Step 8 - Calculate
    va3 = gamma * va2
    # Step 9 - Shape out (N,D)
    out = va3 + beta

    # Calculate running mean and variance to be used on prediction
    running_mean = momentum * running_mean + (1.0 - momentum) * mu
    running_var = momentum * running_var + (1.0 - momentum) * var
    # Store values
    cache = (mu, xmu, carre, var, sqrtvar, invvar,
             va2, va3, gamma, beta, x, bn_param)
  elif mode == 'test':
    # On prediction get the running mean/variance
    running_mean =  bn_param['running_mean']
    running_var  =  bn_param['running_var']
    xbar = (x - running_mean)/np.sqrt(running_var+eps)
    out = gamma*xbar + beta
    cache = (x, xbar, gamma, beta, eps)
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
  # Save updated running mean/variance
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var
  # Return outputs
  return out, cache

def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.
  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  # Recover dropout parameters (p, mask , mode) from cache
  dropout_param, mask = cache
  mode = dropout_param['mode']

  dx = None
  # Back propagate (Dropout layer has no parameters just input X)
  if mode == 'train':
    # Just back propagate dout from the neurons that were used during dropout
    dx = dout * mask
  elif mode == 'test':
    # Disable dropout during prediction/test
    dx = dout

  # Return dx
  return dx

def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.
  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys: (p,test/train,seed)
  Outputs: (out, cache)
  """
  # Get the current dropout mode, p, and seed
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  # Inititalization of outputs and mask
  mask = None
  out = None

  if mode == 'train':
    # Create an apply mask (normally p=0.5 for half of neurons), we scale all
    # by p to avoid having to multiply by p on backpropagation, this is called
    # inverted dropout
    mask = (np.random.rand(*x.shape) < p) / p
    # Apply mask
    out = x * mask
  elif mode == 'test':
    # During prediction no mask is used
    mask = None
    out = x

  # Save mask and dropout parameters for backpropagation
  cache = (dropout_param, mask)

  # Convert "out" type and return output and cache
  out = out.astype(x.dtype, copy=False)
  return out, cache


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.
  Inputs:
  - x: Input Scores (predicted scores from the model) shape (N,Cl)
  - y: Correct scores (Training set correct labels) shape (N,)
  N: Batch size
  Cl: number of classes
  Returns a tuple of: (loss, dout)
  """
  # Just fix numerical instability
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  # Get probabilities and normalize
  probs /= np.sum(probs, axis=1, keepdims=True)

  # N will be the batch size
  N = x.shape[0]

  # Calculate loss
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N

  # Calculate dout gradient (How loss change with respect to x)
  dout_x = probs.copy()
  dout_x[np.arange(N), y] -= 1
  # Scale gradient with relation to N
  dout_x /= N

  # Return loss and dout (Loss gradient with respect to x)
  return loss, dout_x

def relu_backward(dout, cache):
  """
  Computes the backard pass for ReLU
  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Previous input (used on forward propagation)

  Returns:
  - dx: Gradient with respect to x
  """

  # Inititalize dx with None and x with cache
  dx, x = None, cache

  # Make all positive elements in x equal to dout while all the other elements
  # Become zero
  dx = dout * (x >= 0)

  # Return dx (gradient with respect to x)
  return dx

def relu_forward(x):
  """
  Computes the forward pass for ReLU
  Input:
  - x: Inputs, of any shape

  Returns a tuple of: (out, cache)
  The shape on the output is the same as the input
  """
  out = None

  # Create a function that receive x and return x if x is bigger
  # than zero, or zero if x is negative
  relu = lambda x: x * (x > 0).astype(float)
  out = relu(x)

  # Cache input and return outputs
  cache = x
  return out, cache

def fc_forward(x,w,b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    Inputs:
    - x: Input Tensor (N, d_1, ..., d_k)
    - w: Weights (D, M)
    - b: Bias (M,)

    N: Mini-batch size
    M: Number of outputs of fully connected layer
    D: Input dimension
    d_1, ..., d_k: Single input dimension

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None

    #  Get batch size (first dimension)
    N = x.shape[0]

    # Reshape activations to [Nx(d_1, ..., d_k)], which will be a 2d matrix
    # [NxD]
    reshaped_input = x.reshape(N, -1)

    # Calculate output
    out = np.dot(reshaped_input,w) + b.T

    # Save inputs for backward propagation
    cache = (x,w,b)

    # Return outputs
    return out, cache

def fc_backward(dout, cache):
    """
    Computes the backward pass for an affine (fully-connected) layer.

    Inputs:
    - dout: Layer partial derivative wrt loss of shape (N, M) (Same as output)
    - cache: (x,w,b) inputs from previous forward computation

    N: Mini-batch size
    M: Number of outputs of fully connected layer
    D: Input dimension
    d_1, ..., d_k: Single input dimension

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    #  Get batch size (first dimension)
    N = x.shape[0]

    # Get dX (Same format as x)
    dx = np.dot(dout,w.T)
    dx = dx.reshape(x.shape)

    # Get dW (Same format as w)
    # Reshape activations to [Nx(d_1, ..., d_k)], which will be a 2d matrix
    # [NxD]
    reshaped_input = x.reshape(N, -1)
    # Transpose then dot product with dout
    dw = reshaped_input.T.dot(dout)

    # Get dB (Same format as b)
    db = np.sum(dout, axis=0)

    # Return outputs
    return dx, dw, db

def conv_forward_naive(x, w, b, conv_param):
  """
  Computes the forward pass for the Convolution layer. (Naive)
  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': How much pixels the sliding window will travel
    - 'pad': The number of pixels that will be used to zero-pad the input.

  N: Mini-batch size
  C: Input depth (ie 3 for RGB images)
  H/W: Image height/width
  F: Number of filters on convolution layer (Will be the output depth)
  HH/WW: Kernel Height/Width

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape

  # Get parameters
  P = conv_param["pad"]
  S = conv_param["stride"]

  # Calculate output size, and initialize output volume
  H_R = 1 + (H + 2 * P - HH) / S
  W_R = 1 + (W + 2 * P - WW) / S
  out = np.zeros((N,F,H_R,W_R))

  # Pad images with zeros on the border (Used to keep spatial information)
  x_pad = np.lib.pad(x,((0,0),(0,0), (P,P), (P,P)), 'constant', constant_values=0)

  # Apply the convolution
  for n in xrange(N): # For each element on batch
    for depth in xrange(F): # For each filter
      for r in xrange(0,H,S): # Slide vertically taking stride into account
        for c in xrange(0,W,S): # Slide horizontally taking stride into account
          out[n,depth,r/S,c/S] = np.sum(x_pad[n,:,r:r+HH,c:c+WW] * w[depth,:,:,:]) + b[depth]

  # Cache parameters and inputs for backpropagation and return output volume
  cache = (x, w, b, conv_param)
  return out, cache

def conv_backward_naive(dout, cache):
  """
  Computes the backward pass for the Convolution layer. (Naive)
  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
  Returns a tuple of: (dw,dx,db) gradients
  """
  dx, dw, db = None, None, None
  x, w, b, conv_param = cache
  N, F, H_R, W_R = dout.shape
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  P = conv_param["pad"]
  S = conv_param["stride"]
  # Do zero padding on x_pad
  x_pad = np.lib.pad(x,((0,0),(0,0), (P,P), (P,P)), 'constant', constant_values=0)

  # Inititalize outputs
  dx = np.zeros(x_pad.shape)
  dw = np.zeros(w.shape)
  db = np.zeros(b.shape)

  # Calculate dx, with 2 extra col/row that will be deleted
  for n in xrange(N): # For each element on batch
    for depth in xrange(F): # For each filter
      for r in xrange(0,H,S): # Slide vertically taking stride into account
        for c in xrange(0,W,S): # Slide horizontally taking stride into account
          dx[n,:,r:r+HH,c:c+WW] += dout[n,depth,r/S,c/S] * w[depth,:,:,:]

  #deleting padded rows to match real dx
  delete_rows    = range(P) + range(H+P,H+2*P,1)
  delete_columns = range(P) + range(W+P,W+2*P,1)
  dx = np.delete(dx, delete_rows, axis=2)     #height
  dx = np.delete(dx, delete_columns, axis=3)  #width

  # Calculate dw
  for n in xrange(N): # For each element on batch
    for depth in xrange(F): # For each filter
      for r in xrange(H_R): # Slide vertically taking stride into account
        for c in xrange(W_R): # Slide horizontally taking stride into account
          dw[depth,:,:,:] += dout[n,depth,r,c] * x_pad[n,:,r*S:r*S+HH,c*S:c*S+WW]

  # Calculate db, 1 scalar bias per filter, so it's just a matter of summing
  # all elements of dout per filter
  for depth in range(F):
    db[depth] = np.sum(dout[:, depth, :, :])

  return dx, dw, db

def max_pool_forward_naive(x, pool_param):
  """
  Compute the forward propagation of max pooling (naive way)
  Inputs:
  - x: 4d Input tensor , of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_heigh/widtht': Sliding window height/width
    - 'stride': Sliding moving distance
  N: Mini-batch size
  C: Input depth (ie 3 for RGB images)
  H/W: Image height/width
  HH/WW: Kernel Height/Width

  Returns a tuple of: (out, cache)
  """
  # Get input tensor and parameter data
  N, C, H, W = x.shape
  S = pool_param["stride"]
  # Consider H_P and W_P as the sliding window height and width
  H_P = pool_param["pool_height"]
  W_P = pool_param["pool_width"]

  # Calculate output size
  out = None
  HH = 1 + (H - H_P) / S
  WW = 1 + (W - W_P) / S
  out = np.zeros((N,C,HH,WW))

  # Calculate output (Both for loops do the same thing ....)
  #for n in xrange(N): # For each element on batch
    #for depth in xrange(C): # For each input depth
      #for r in xrange(HH): # Slide vertically
        #for c in xrange(WW): # Slide horizontally
          # Get biggest element on the window
          #out[n,depth,r,c] = np.max(x[n,depth,r*S:r*S+H_P,c*S:c*S+W_P])

  # Calculate output
  for n in xrange(N): # For each element on batch
    for depth in xrange(C): # For each input depth
      for r in xrange(0,H,S): # Slide vertically taking stride into account
        for c in xrange(0,W,S): # Slide horizontally taking stride into account
          # Get biggest element on the window
          out[n,depth,r/S,c/S] = np.max(x[n,depth,r:r+H_P,c:c+W_P])

  # Return output and save inputs and paramters to cache
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  Compute the backward propagation of max pooling (naive way)
  Inputs:
  - dout: Upstream derivatives, same size as cached x
  - cache: A tuple of (x, pool_param) as in the forward pass.
  Returns:
  - dx: Gradient with respect to x
  """
  # Get data back from cache
  x, pool_param = cache

  # Get input tensor and parameter
  N, C, H, W = x.shape
  S = pool_param["stride"]
  H_P = pool_param["pool_height"]
  W_P = pool_param["pool_width"]
  N,C,HH,WW = dout.shape

  # Inititalize dx
  dx = None
  dx = np.zeros(x.shape)

  # Calculate dx (mask * dout)
  for n in xrange(N): # For each element on batch
    for depth in xrange(C): # For each input depth
      for r in xrange(HH): # Slide vertically (use stride on the fly)
        for c in xrange(WW): # Slide horizontally (use stride on the fly)
          # Get window and calculate the mask
          x_pool = x[n,depth,r*S:r*S+H_P,c*S:c*S+W_P]
          mask = (x_pool == np.max(x_pool))
          # Calculate mask*dout
          dx[n,depth,r*S:r*S+H_P,c*S:c*S+W_P] = mask*dout[n,depth,r,c]

  # Return dx
  return dx
