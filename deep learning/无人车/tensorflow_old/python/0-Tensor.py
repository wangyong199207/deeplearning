
# coding: utf-8

# ### Initialization

# In[ ]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

sess = tf.InteractiveSession()


# ### Create tensors

# In[ ]:

t0 = tf.constant(1)
t1 = tf.constant([1, 2])
t2 = tf.constant([[1, 2], [3, 4], [5, 6]])
t3 = tf.constant([[[1], [2]], [[3], [4]], [[5], [6]]])


# In[ ]:

print t0.eval()


# In[ ]:

print t1.eval()


# In[ ]:

print t2.eval()


# In[ ]:

print t3.eval()


# ### Tensor rank: number of dimensions

# In[ ]:

print 'Rank t0: {}'.format(tf.rank(t0).eval())
print 'Rank t1: {}'.format(tf.rank(t1).eval())
print 'Rank t2: {}'.format(tf.rank(t2).eval())
print 'Rank t3: {}'.format(tf.rank(t3).eval())


# ### Tensor shape: size of each dimension

# In[ ]:

print 'Shape t0: {}'.format(tf.shape(t0).eval())
print 'Shape t1: {}'.format(tf.shape(t1).eval())
print 'Shape t2: {}'.format(tf.shape(t2).eval())
print 'Shape t3: {}'.format(tf.shape(t3).eval())


# ### Tensor size: number of elements

# In[ ]:

print 'Size t0: {}'.format(tf.size(t0).eval())
print 'Size t1: {}'.format(tf.size(t1).eval())
print 'Size t2: {}'.format(tf.size(t2).eval())
print 'Size t3: {}'.format(tf.size(t3).eval())


# ### Tensor reshape

# In[ ]:

print t3.eval()


# In[ ]:

t3_r = tf.reshape(t3, [2, 3])
print t3_r.eval()


# In[ ]:

print tf.shape(t3_r).eval()


# In[ ]:

print tf.rank(t3_r).eval()


# In[ ]:

print tf.size(t3_r).eval()


# In[ ]:

# flatten a tensor
print tf.reshape(t3, [-1]).eval()


# In[ ]:

# infer the shape
print tf.reshape(t3, [2, -1]).eval()


# ### Tensor squeeze: remove dimension of size 1

# In[ ]:

print tf.squeeze(t3).eval()


# ### Homework

# In[ ]:

# Other tensor transformations
# https://www.tensorflow.org/versions/r0.11/api_docs/python/array_ops.html

# expand_dims(), slice(), pad(), concat(), ...


# In[ ]:



