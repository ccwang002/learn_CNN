"""Benchmark for two-layer NN model.

The NN model has the following structure::

    [input_size] --> [hidden_size] --> [num_classes]
    [     D    ]     [     H     ]     [     C     ]
               W1, b1            W2, b2

D, H, C are the notations for their dimension sizes.
If rand_seed is set, generated model should be the same.

Common Parameters
-----------------
input_size : D
hidden_size : H
num_classes : C
num_inputs : N
rand_seed : any object (usually int), None
"""
import numpy as np


def init_benchmark_model(
    input_size, hidden_size, num_classes, rand_seed=None,
    **kwargs
):
    """Initiate weights and biases of a two-layer NN model."""
    rs = np.random.RandomState(seed=rand_seed)

    model = {}
    D, H, C = input_size, hidden_size, num_classes
    model['W1'] = rs.rand(D, H)
    model['b1'] = rs.rand(H)
    model['W2'] = rs.rand(H, C)
    model['b2'] = rs.rand(C)

    return model


def init_benchmark_data(
    num_inputs, input_size, num_classes, rand_seed=None,
    **kwargs
):
    """Generate randomized data for benchmark."""
    N, D, C = num_inputs, input_size, num_classes

    rs = np.random.RandomState(seed=rand_seed)
    X = rs.rand(N, D)
    y = rs.choice(C, size=N)
    return X, y
