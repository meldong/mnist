# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:40:40 2017

@author: kd
"""

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

print("----------------------------------------------------------------------")
print("0. Begin the show")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("")

print("----------------------------------------------------------------------")
print("1. 來看看 MNIST 的型態")
print(type(mnist))
print(mnist.train.num_examples)
print(mnist.validation.num_examples)
print(mnist.test.num_examples)
print("")

print("----------------------------------------------------------------------")
print("2. 讓我們看一下 MNIST 訓練還有測試的資料集長得如何")
print("   train_img 的 type : %s" % (type(mnist.train.images)))
print("   train_img 的 dimension : %s" % (mnist.train.images.shape,))
print("   train_label 的 type : %s" % (type(mnist.train.labels)))
print("   train_label 的 dimension : %s" % (mnist.train.labels.shape,))
print("   test_img 的 type : %s" % (type(mnist.test.images)))
print("   test_img 的 dimension : %s" % (mnist.test.images.shape,))
print("   test_label 的 type : %s" % (type(mnist.test.labels)))
print("   test_label 的 dimension : %s" % (mnist.test.labels.shape,))
print("")

print("----------------------------------------------------------------------")
print("3. 讓我們看一下 MNIST 實際印出會長怎麼樣")
nsample = 1
randidx = np.random.randint(mnist.train.images.shape[0], size = nsample)
for i in range(nsample):
    img = np.reshape(mnist.train.images[randidx[i], :], (28, 28)) # matrix
    label = np.argmax(mnist.train.labels[randidx[i], :] ) # label
    plt.imshow(img, interpolation='lanczos', cmap=plt.get_cmap('gray'))
    plt.title("No. " + str(randidx[i] + 1) + " Training Data " + "Label is " + str(label))
    plt.savefig(str(randidx[i] + 1) + "_Label_" + str(label) +".jpg")
    plt.show()
print("")

print("----------------------------------------------------------------------")
print("4. 讓我們實作 MNIST Softmax 模型")
import tensorflow as tf

# Feed the data
x_data = tf.placeholder(tf.float32, [None, 784])
y_data = tf.placeholder(tf.float32, [None, 10])

# Build the graph
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x_data, W) + b

# Activate the output
y = tf.nn.softmax(y)

# Define the loss and optimizer
loss = tf.reduce_mean(-tf.reduce_sum(y_data * tf.log(y), reduction_indices=[1])) # cross entropy
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
# optimizer = tf.train.AdamOptimizer(1e-4)
train = optimizer.minimize(loss)

# Define the accuracy
correct_prediction = tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_data, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

# Initialize the variables
init = tf.global_variables_initializer()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x_data: batch_xs, y_data: batch_ys})
    if i % 100 == 0:
        print('step %d, training accuracy %g' % (i, sess.run(accuracy, feed_dict = {x_data: batch_xs, y_data: batch_ys})))
        
# Test trained model
print('testing accuracy is %g' % sess.run(accuracy, feed_dict = {x_data: mnist.test.images, y_data: mnist.test.labels}))
print(sess.run(y[0,:], feed_dict = {x_data: mnist.test.images, y_data: mnist.test.labels}))
sess.close()
print("")

print("----------------------------------------------------------------------")
print("4. 讓我們實作 MNIST Softmax 模型")
import tensorflow as tf

# Feed the data
x_data = tf.placeholder(tf.float32, [None, 784])
y_data = tf.placeholder(tf.float32, [None, 10])

# Build the graph
def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = 0.5
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

y = deepnn(x_data)

# Activate the output
# y = tf.nn.softmax(y)

# Define the loss and optimizer
# loss = tf.reduce_mean(-tf.reduce_sum(y_data * tf.log(y), reduction_indices=[1])) # cross entropy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=y))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
optimizer = tf.train.AdamOptimizer(1e-4)
train = optimizer.minimize(loss)

# Define the accuracy
correct_prediction = tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_data, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

# Initialize the variables
init = tf.global_variables_initializer()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x_data: batch_xs, y_data: batch_ys})
    if i % 100 == 0:
        print('step %d, training accuracy %g' % (i, sess.run(accuracy, feed_dict = {x_data: batch_xs, y_data: batch_ys})))
        
# Test trained model
print('testing accuracy is %g' % sess.run(accuracy, feed_dict = {x_data: mnist.test.images, y_data: mnist.test.labels}))
print(sess.run(tf.nn.softmax(y[0,:]), feed_dict = {x_data: mnist.test.images, y_data: mnist.test.labels}))
sess.close()
print("")
