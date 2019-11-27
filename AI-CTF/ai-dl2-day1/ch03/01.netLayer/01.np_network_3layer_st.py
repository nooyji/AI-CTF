# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:04:33 2018

@author: shkim
"""
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def identity_func(x):
    return x

# network : input(2) -> hidden(3) -> hidden(2) -> output(2)
# 2x*w1+b1 -> 3z1*w2+b2 -> 2z2*w3+b3 -> 2y
def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])    
    network['W2'] = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])    
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3], [0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    return network

def forward(network, x):
    W1, W2, W3 = ........
    b1, b2, b3 = ........
    
    a1 = ........
    z1 = ........
    a2 = ........
    z2 = ........
    a3 = ........
    y = .........
    
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(......)
print(y)    # [0.31682708 0.69627909]


    