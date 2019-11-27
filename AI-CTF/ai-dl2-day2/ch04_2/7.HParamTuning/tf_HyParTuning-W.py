# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 12:25:38 2019

@author: shkim
"""

import numpy as np
import tensorflow as tf

#%%
# case-1
#%%
w = tf.Variable(0., dtype=tf.float32)

# Lf = (w - 5)**2  --> Lf = (w**2 - 10*w + 25)
# Lf --> minimize --> w:5
#cost = tf.add(tf.add(w**2, tf.multiply(-10., w)), 25)
cost = w**2 - 10*w + 25
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(w))

#%%
sess.run(train)
print(sess.run(w))

#%%
for i in range(1000):
    sess.run(train)
    
print(sess.run(w))

#%%
# case-2
#%%
#coefficients = np.array([[1.], [-10.], [25.]])
coefficients = np.array([[1.], [-20.], [100.]])  # case-3
w = tf.Variable(0., dtype=tf.float32)
x = tf.placeholder(tf.float32, [3,1])

# Lf = (w - 5)**2  --> Lf = (w**2 - 10*w + 25)
# Lf --> minimize --> w:5
#cost = tf.add(tf.add(w**2, tf.multiply(-10., w)), 25)
#cost = w**2 - 10*w + 25
cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(w))

#%%
sess.run(train, feed_dict={x:coefficients})
print(sess.run(w))

#%%
for i in range(1000):
    sess.run(train, feed_dict={x:coefficients})
    
print(sess.run(w))

#%%
# case-3
coefficients = np.array([[1.], [-20.], [100.]])

#%%
# case-4 just memo

#sess = tf.Session()
#sess.run(init)
#print(sess.run(w))

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(w))

#%%
print(np.random.rand())