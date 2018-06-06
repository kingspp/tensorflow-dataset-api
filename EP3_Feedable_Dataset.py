# -*- coding: utf-8 -*-
"""
| **@created on:** 06/06/18,
| **@author:** Prathyush SP,
| **@version:** v0.0.1
|
| **Description:**
| Feedable Dataset
|
| **Sphinx Documentation Status:** Complete
|
..todo::
"""

# Imports
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

start = time.time()

# Global Variables
EPOCH = 10
BATCH_SIZE = 32
DISPLAY_STEP = 1

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Create Placeholders
features_placeholder = tf.placeholder(mnist.train.images.dtype, [None, mnist.train.images.shape[-1]])
labels_placeholder = tf.placeholder(mnist.train.labels.dtype, [None, mnist.train.labels.shape[-1]])

# Create Tensor slices from placeholders
train_dataset = tf.data.Dataset.from_tensor_slices(features_placeholder)
label_dataset = tf.data.Dataset.from_tensor_slices(labels_placeholder)  # .map(lambda z: tf.one_hot(z, 10))

# Create Dataset
dataset = tf.data.Dataset.zip((train_dataset, label_dataset)).batch(BATCH_SIZE).repeat(EPOCH)

# Create Dataset Iterator
iterator = dataset.make_initializable_iterator()

# Create features and labels
features, labels = iterator.get_next()


# Deeplearning Model
def nn_model(features, labels):
    bn = tf.layers.batch_normalization(features)
    fc1 = tf.layers.dense(bn, 50)
    fc2 = tf.layers.dense(fc1, 50)
    fc2 = tf.layers.dropout(fc2)
    fc3 = tf.layers.dense(fc2, 10)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fc3))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    return optimizer, loss


# Create elements from iterator
training_op, loss_op = nn_model(features=features, labels=labels)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    sess.run(iterator.initializer, feed_dict={features_placeholder: mnist.train.images,
                                              labels_placeholder: mnist.train.labels})
    batch_id, epoch_id, total_batches, avg_cost = 0, 0, int(mnist.train.num_examples / BATCH_SIZE), 0
    while not sess.should_stop():
        _, c = sess.run([training_op, loss_op])
        avg_cost += c / total_batches
        if batch_id == total_batches:
            if epoch_id % DISPLAY_STEP == 0:
                print("Epoch:", '%04d' % (epoch_id + 1), "cost={:.9f}".format(avg_cost))
            batch_id, avg_cost, cost = 0, 0, []
            epoch_id += 1
        batch_id += 1
    print("Optimization Finished!")

print('Total Time Elapsed: {} secs'.format(time.time() - start))
