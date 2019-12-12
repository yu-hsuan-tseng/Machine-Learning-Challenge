'''
    machine learning 
    day 94
'''
# hyperparameter modification 
# padding is for the same size convolutional

import numpy as np

def conv_single_step(a_slice_prev,w,b):
    s = a_slice_prev * w
    z = np.sum(s)
    z = float(z+b)
    return z

np.random.seed(1)
a_slice_prev = np.random.randn(4,4,3)
w = np.random.randn(4,4,3)
b = np.random.randn(1,1,1)

z = conv_single_step(a_slice_prev,w,b)
print("z",z)