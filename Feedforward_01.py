# author: JinyuZ1996
# Creation data: 2020/08/24

import numpy as np

x = np.array([0.5, 0.4, 0.3, 0.4, 0.5])
w = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
A = np.dot(x, w)

print(w)
print(A)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

fade_1 = sigmoid(A)
print(fade_1)

w2 = np.array([[3, 2], [3, 2], [2, 3], [1, 1]])
Y = np.dot(fade_1, w2)
print(Y)

def softmax(z):
    if z.ndim == 2:
        z = z.T
        z = z - np.max(z, axis=0)
        y = np.exp(z) / np.sum(np.exp(z), axis=0)
        return y.T
    else:
        y = np.exp(z) / np.sum(np.exp(z), axis=0)
        return y.T

print(softmax(Y))
