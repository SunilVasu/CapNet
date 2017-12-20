from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import tempfile


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
y_ = tf.placeholder(shape=[None, 10], dtype=tf.float32)

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

graph_location = tempfile.mkdtemp()
print('Saving grap to %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())

n_epochs = 10
batch_size = 50
restore_checkpoint = True

n_iterations_per_epoch = mnist.train.num_examples // batch_size
n_iterations_validation = mnist.validation.num_examples // batch_size
best_loss_val = np.infty
checkpoint_path = "./my_convnet1"

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def train():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
            saver.restore(sess, checkpoint_path)
        else:
            init.run()
        for epoch in range(n_epochs):
            for iteration in range(1, n_iterations_per_epoch + 1):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                _, loss_train = sess.run(
                    [train_step, cross_entropy],
                    feed_dict={x: X_batch.reshape([-1, 28, 28, 1]),
                               y_: y_batch,
                               keep_prob: 0.5})
                print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                    iteration, n_iterations_per_epoch,
                    iteration * 100 / n_iterations_per_epoch,
                    loss_train),
                    end="")

            loss_vals = []
            acc_vals = []
            for iteration in range(1, n_iterations_validation + 1):
                X_batch, y_batch = mnist.validation.next_batch(batch_size)
                loss_val, acc_val = sess.run(
                    [cross_entropy, accuracy],
                    feed_dict={x: X_batch.reshape([-1, 28, 28, 1]),
                               y_: y_batch,
                               keep_prob: 1.0})
                loss_vals.append(loss_val)
                acc_vals.append(acc_val)
                print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                    iteration, n_iterations_validation,
                    iteration * 100 / n_iterations_validation),
                    end=" " * 10)
            loss_val = np.mean(loss_vals)
            acc_val = np.mean(acc_vals)
            print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
                epoch + 1, acc_val * 100, loss_val,
                " (improved)" if loss_val < best_loss_val else ""))

            if loss_val < best_loss_val:
                save_path = saver.save(sess, checkpoint_path)
                best_loss_val = loss_val


def test():

    n_iterations_test = mnist.test.num_examples // batch_size

    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)

        loss_tests = []
        acc_tests = []
        for iteration in range(1, n_iterations_test + 1):
            X_batch, y_batch = mnist.test.next_batch(batch_size)
            loss_test, acc_test = sess.run(
                [cross_entropy, accuracy],
                feed_dict={x: X_batch.reshape([-1, 28, 28, 1]),
                           y_: y_batch,
                           keep_prob: 1.0})
            loss_tests.append(loss_test)
            acc_tests.append(acc_test)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                iteration, n_iterations_test,
                iteration * 100 / n_iterations_test),
                end=" " * 10)
        loss_test = np.mean(loss_tests)
        acc_test = np.mean(acc_tests)
        print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
            acc_test * 100, loss_test))
