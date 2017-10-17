# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:40:40 2017

@author: aenldong
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
nsample = 55000
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

# Create the model
x_data = tf.placeholder(tf.float32, [None, 784])
y_data = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x_data, W) + b)

# Define the loss and optimizer
loss = tf.reduce_mean(-tf.reduce_sum(y_data * tf.log(y), reduction_indices=[1])) # cross entropy
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Initialize the variables
init = tf.global_variables_initializer()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x_data: batch_xs, y_data: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_data, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)
print(sess.run(accuracy, feed_dict = {x_data: mnist.test.images, y_data: mnist.test.labels}))
print(sess.run(y[0,:], feed_dict = {x_data: mnist.test.images, y_data: mnist.test.labels}))
