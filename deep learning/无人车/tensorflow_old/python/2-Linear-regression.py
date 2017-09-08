
# coding: utf-8

# ### Linear regression: fit a line y=2*x+3 

# In[ ]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:

# Training Data
train_X = np.linspace(-1, 1, 50)
train_Y = 2 * train_X + 3 + np.random.randn(*train_X.shape) * 0.33
n_samples = train_X.shape[0]

print train_X
print train_Y
print n_samples


# In[ ]:

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")


# In[ ]:

# Construct a linear model
pred = tf.add(tf.mul(X, W), b)


# In[ ]:

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# In[ ]:

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    # show the initial data
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.legend()
    plt.show()
    _ = raw_input("Press [enter] to continue.")
    plt.close()
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    _ = raw_input("Press [enter] to continue.")
    
    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),                 "W=", sess.run(W), "b=", sess.run(b)

    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'

    #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()


# In[ ]:



