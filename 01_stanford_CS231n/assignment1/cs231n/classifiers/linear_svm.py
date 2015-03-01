import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops)
    Inputs:
    - W: C x D array of weights
    - X: D x N array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)   # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[0]
    num_train = X.shape[1]
    loss = 0.0
    for i in range(num_train):
        X_i = X[:, i]
        scores = W.dot(X_i)
        correct_class_score = scores[y[i]]
        num_fail_margin_classes = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1   # note delta = 1
            if margin > 0:
                loss += margin
                num_fail_margin_classes += 1
                # update the gradient
                dW[j] += X_i

        # update gradient of y[i] (ground truth class)
        dW[y[i]] -= num_fail_margin_classes * X_i

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    # Now dW is the sum of all samples' gradient
    dW /= num_train
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    num_classes = W.shape[0]
    num_train = X.shape[1]
    loss = 0.0
    # dW = np.zeros(W.shape)  # initialize the gradient as zero

    _train_ix = np.arange(num_train)
    scores = W.dot(X)       # scores in shape C x N
    correct_class_scores = scores[y, _train_ix]  # shape: N
    margins = scores - correct_class_scores + 1  # delta = 1, shape C x N
    positive_margins_mask = margins > 0
    loss = np.sum(margins[positive_margins_mask]) / num_train

    # since correct class is included in margins, whose margin will always be 1
    loss -= 1
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    multiplier = positive_margins_mask.astype(np.int).T   # shape: N x C
    failed_margin_count = np.sum(positive_margins_mask, axis=0)  # shape: N
    multiplier[_train_ix, y] -= failed_margin_count

    dW = (X.dot(multiplier)).T
    dW /= num_train

    return loss, dW
