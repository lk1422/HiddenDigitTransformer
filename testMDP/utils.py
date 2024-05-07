import numpy as np

def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=0, keepdims=True)
