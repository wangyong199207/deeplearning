
# coding: utf-8

# # Const, Variable, and Placeholder

# In[ ]:

import tensorflow as tf


# ### Const
# - Basic operations

# In[ ]:

a = tf.constant(2)
b = tf.constant(3)
print a


# In[ ]:

# Launch the default graph.
with tf.Session() as sess:
    print "a=2, b=3"
    print "Addition with constants: %i" % sess.run(a+b)
    print "Multiplication with constants: %i" % sess.run(a*b)


# - Matrix operations

# In[ ]:

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1, matrix2)


# In[ ]:

with tf.Session() as sess:
    result = sess.run(product)
    print result


# ### Placeholder

# In[ ]:

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)


# In[ ]:

add = tf.add(a, b)
mul = tf.mul(a, b)


# In[ ]:

with tf.Session() as sess:
    # Run every operation with variable input
    print "Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3})
    print "Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3})


# ### Variable

# In[ ]:

# Create a Variable, that will be initialized to the scalar value 0.
state = tf.Variable(0, name="counter")

# Create an Op to add one to `state`.
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# Variables must be initialized by running an `init` Op after having
# launched the graph.
init_op = tf.initialize_all_variables()

# Launch the graph and run the ops.
with tf.Session() as sess:
    with tf.device("/cpu:0"):
        sess.run(init_op)
        print(sess.run(state))
        # Run the op that updates 'state' and print 'state'.
        for _ in range(3):
            sess.run(update)
            print(sess.run(state))


# ### Homework

# In[ ]:

# other math ops
# https://www.tensorflow.org/versions/r0.11/api_docs/python/math_ops.html
# add(), sub(), mul(), div(), exp(), log()

