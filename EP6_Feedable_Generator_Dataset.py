# -*- coding: utf-8 -*-
"""
| **@created on:** 06/06/18,
| **@author:** Prathyush SP,
| **@version:** v0.0.1
|
| **Description:**
| Feedable Generator Dataset
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


def input_train_gen_fn():
    for d in mnist.train.images:
        yield d


def input_label_gen_fn():
    for d in mnist.train.labels:
        yield d


# Create Placeholders
features_placeholder = tf.placeholder(tf.float32, [None, mnist.train.images.shape[-1]], name='fpl')
labels_placeholder = tf.placeholder(tf.float32, [None, mnist.train.labels.shape[-1]], name='lpl')


# Deeplearning Model
def nn_model(features, labels):
    bn = tf.layers.batch_normalization(features)
    fc1 = tf.layers.dense(bn, 50)
    fc2 = tf.layers.dense(fc1, 50)
    fc2 = tf.layers.dropout(fc2)
    fc3 = tf.layers.dense(fc2, 10)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fc3), name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    return optimizer, loss


# Create elements from iterator
training_op, loss_op = nn_model(features=features_placeholder, labels=labels_placeholder)
global_init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()
graph_def = tf.get_default_graph().as_graph_def()
tf.reset_default_graph()


# Create Tensor slices from placeholders
train_dataset = tf.data.Dataset.from_generator(input_train_gen_fn, tf.float32)
label_dataset = tf.data.Dataset.from_generator(input_label_gen_fn, tf.float32)

# Create Dataset
dataset = tf.data.Dataset.zip((train_dataset, label_dataset)).batch(BATCH_SIZE).repeat(EPOCH)

# Create Dataset Iterator
iterator = dataset.make_one_shot_iterator()

# Create features and labels
features, labels = iterator.get_next()

[loss_op, gi, li, training_op] = tf.import_graph_def(graph_def, input_map={'fpl': features, 'lpl': labels},
                                                     return_elements=['loss:0', 'init', 'init_1', 'Adam'])

with tf.train.MonitoredTrainingSession(scaffold=tf.train.Scaffold(init_op=tf.group(gi, li))) as sess:
    # sess.run(tf.group(gi, li))
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
