import numpy as np
# from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    Inputs:
    - W: C x D array of weights
    - X: D x N array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W, an array of same size as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[1]
    f = W.dot(X)  # shape: C x N
    p = np.zeros(num_train, dtype=np.float)

    for i in range(num_train):
        f_i = f[:, i].copy()  # shape C x 1
        f_i -= np.max(f_i)
        f_i = np.exp(f_i)
        x_i = X[:, i]
        all_class_p_i = f_i / np.sum(f_i)
        p[i] = all_class_p_i[y[i]]

        # Update gradient
        # all_class_p_i no used later, don't copy
        dw_x_weight_i = all_class_p_i
        dw_x_weight_i[y[i]] -= 1
        dW -= dw_x_weight_i[:, np.newaxis] * x_i[np.newaxis, :]

    loss += np.mean(-np.log(p))
    # Add regularization
    loss += 0.5 * reg * np.sum(W * W)

    # Gradient
    # ref: http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
    dW /= -num_train
    dW += reg * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[1]
    _train_ix = np.arange(num_train)  # for sample coord 0...N-1

    f = W.dot(X)  # shape: C x N
    f -= np.max(f, axis=0)
    f = np.exp(f)
    p = f / np.sum(f, axis=0)  # shape: C x N

    # loss function
    loss += np.mean(-np.log(p[y, _train_ix]))
    loss += 0.5 * reg * np.sum(W * W)

    # gradient
    dW_x_weight = p  # no use p later, don't copy
    dW_x_weight[y, _train_ix] -= 1
    # CxD -= CxN dot NxD
    dW -= dW_x_weight.dot(X.T)
    dW /= -num_train
    dW += reg * W

    return loss, dW
