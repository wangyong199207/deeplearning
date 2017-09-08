
# coding: utf-8

# ### MNIST dataset
# http://yann.lecun.com/exdb/mnist/
# ![alt text](http://neuralnetworksanddeeplearning.com/images/mnist_100_digits.png "MNIST")
# 

# In[ ]:

# Import MNIST
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix


# In[ ]:

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("./MNIST_data", one_hot=True)


# In[ ]:

img = data.train.images[10, :].reshape((28,28))
plt.imshow(img, cmap='gray')


# In[ ]:

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of classes, one class for each of 10 digits.
num_classes = 10


# In[ ]:

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])


# In[ ]:

data.test.cls = np.array([label.argmax() for label in data.test.labels])

# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)


# In[ ]:

data.test.labels[0,:]


# In[ ]:



