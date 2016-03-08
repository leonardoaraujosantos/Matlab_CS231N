# Reference
# http://code.activestate.com/recipes/521906-k-fold-cross-validation-partition/
# http://stats.stackexchange.com/questions/95797/how-to-split-the-dataset-for-cross-validation-learning-curve-and-final-evaluat
def k_fold_cross_validation(X, K, randomise = False):
    """
    Generates K (training, validation) pairs from the items in X.

    Each pair is a partition of X, where validation is an iterable
    of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

    If randomise is true, a copy of X is shuffled before partitioning,
    otherwise its order is preserved in training and validation.
    """
    if randomise: from random import shuffle; X=list(X); shuffle(X)
    for k in xrange(K):
        training = [x for i, x in enumerate(X) if i % K != k]
        validation = [x for i, x in enumerate(X) if i % K == k]
        # Returns a generator (to be used on the for-loops)         
        yield training, validation
        
def test_k_fold(X,K):
    for training, validation in k_fold_cross_validation(X, K):
        1+1
    return [training, validation]            

# Example
# X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
# for training, validation in k_fold_cross_validation(X, K=5):
#     for x in X: assert (x in training) ^ (x in validation), x
