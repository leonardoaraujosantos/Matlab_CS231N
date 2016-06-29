import numpy as np

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
