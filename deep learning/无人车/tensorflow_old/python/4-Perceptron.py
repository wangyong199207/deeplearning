
# coding: utf-8

# ## Perceptron

# In[ ]:

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# ### Load data

# In[ ]:

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


# ### Build graph

# In[ ]:

X = tf.placeholder("float", [None, 784]) # input image
Y = tf.placeholder("float", [None, 10]) # outupt: 10 classes

w_h = tf.Variable(tf.random_normal([784, 625], stddev=0.01)) # hidden layer: 625 neurons
w_o = tf.Variable(tf.random_normal([625, 10], stddev=0.01)) # output layer
tf.histogram_summary("w_h", w_h)
tf.histogram_summary("w_o", w_o)

h = tf.nn.sigmoid(tf.matmul(X, w_h))
py_x = tf.matmul(h, w_o)

with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
    tf.scalar_summary("cost", cost)

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1)) # Count correct predictions
    acc = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average
    tf.scalar_summary("accuracy", acc)


# ### Launch graph
# - After launch graph, run `tensorboard --logdir=./logs/perceptron` to launch tensorboard
# - Open `http://localhost:6006` in brower

# In[ ]:

# Launch the graph in a session
with tf.Session() as sess:
    # create a log writer. run 'tensorboard --logdir=./logs/perceptron'
    writer = tf.train.SummaryWriter("./logs/perceptron", sess.graph)
    merged = tf.merge_all_summaries()
    
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    # 100 batches of size 128
    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        summary, accuracy = sess.run([merged, acc], feed_dict={X: teX, Y: teY})
        writer.add_summary(summary, i)
        # print (iter, accuracy)
        print(i, accuracy)


# In[ ]:



