# -*- coding: utf-8 -*-
"""
| **@created on:** 12/06/18,
| **@author:** Prathyush SP,
| **@version:** v0.0.1
|
| **Description:**
| Feedable Iterator Dataset
| **Sphinx Documentation Status:** Complete
|
..todo::
"""
import os
import sys
import time

if len(sys.argv) <= 1:
    sys.argv.append('cpu')
USE_GPU = True if sys.argv[1] == 'gpu' else False
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if USE_GPU else ""

from benchmark.benchmark import BenchmarkUtil
from benchmark.system_monitors import CPUMonitor, MemoryMonitor, GPUMonitor
import random

butil = BenchmarkUtil(
    model_name='EP15 Feedable Iterator, Multiple Dataset, Initializable Iterator {}'.format(sys.argv[1]),
    stats_save_path='/tmp/stats/',
    monitors=[CPUMonitor, MemoryMonitor, GPUMonitor])


# @butil.monitor
def main():
    # Imports
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data
    import time

    start = time.time()

    # Global Variables
    EPOCH = 10
    BATCH_SIZE = 32
    bs_placeholder = tf.placeholder(dtype=tf.int64)
    DISPLAY_STEP = 1

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Create Dataset Handle
    feedable_feature_dataset_handle = tf.placeholder(tf.string, shape=[])
    feedable_dont_care_dataset_handle = tf.placeholder(tf.string, shape=[])
    feedable_label_dataset_handle = tf.placeholder(tf.string, shape=[])

    # Create Train Dataset
    features_dataset = tf.data.Dataset.from_tensor_slices(mnist.train.images).batch(bs_placeholder)
    dontc_care_dataset = tf.data.Dataset.from_tensor_slices(mnist.train.images).batch(bs_placeholder)
    label_dataset = tf.data.Dataset.from_tensor_slices(mnist.train.labels).batch(bs_placeholder)

    # Create Test Dataset
    test_features_dataset = tf.data.Dataset.from_tensor_slices(mnist.test.images)
    test_label_dataset = tf.data.Dataset.from_tensor_slices(mnist.test.labels)
    # test_dataset = tf.data.Dataset.zip((test_features_dataset, test_label_dataset)).batch(
    #     batch_size=mnist.test.num_examples)

    # Create Dataset Iterators
    features_dataset_iterator = features_dataset.make_initializable_iterator()
    dont_care_dataset_iterator = dontc_care_dataset.make_initializable_iterator()
    label_dataset_iterator = label_dataset.make_initializable_iterator()

    # test_iterator = test_dataset.make_initializable_iterator()

    # Create Feedable Iterator 1
    feedable_feature_dataset_iterator = tf.data.Iterator.from_string_handle(
        feedable_feature_dataset_handle, features_dataset.output_types)

    # Create Feedable Iterator 2
    feedable_dont_care_dataset_iterator = tf.data.Iterator.from_string_handle(
        feedable_dont_care_dataset_handle, dontc_care_dataset.output_types)

    # Create Feedable Iterator 2
    feedable_label_dataset_iterator = tf.data.Iterator.from_string_handle(
        feedable_label_dataset_handle, label_dataset.output_types)

    # Create features and labels
    feature_element = feedable_feature_dataset_iterator.get_next()
    dont_care_element = feedable_dont_care_dataset_iterator.get_next()
    label_element = feedable_label_dataset_iterator.get_next()

    # Create Default Placeholder for Features and Labels
    features_p = tf.placeholder_with_default(feature_element, shape=[None, 784])
    dont_care_p = tf.placeholder_with_default(dont_care_element, shape=[None, 784])
    labels_p = tf.placeholder_with_default(label_element, shape=[None, 10])

    # Deeplearning Model
    def nn_model(features, dont_care, labels):
        bn = tf.layers.batch_normalization(features)
        fc1 = tf.layers.dense(bn, 50)
        fc2 = tf.layers.dense(fc1, 50)
        fc2 = tf.layers.dropout(fc2)
        fc3 = tf.layers.dense(fc2, 10)
        l1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=fc3))
        o1 = tf.train.AdamOptimizer(learning_rate=0.01).minimize(l1)

        fc4 = tf.layers.dense(dont_care, 50)
        l2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=fc4))
        o2 = tf.train.AdamOptimizer(learning_rate=0.01).minimize(l2)

        return o1, o2, l1, l2

    # Create Handles
    features_dataset_handle = features_dataset_iterator.string_handle()
    dont_care_dataset_handle = dont_care_dataset_iterator.string_handle()
    label_dataset_handle = label_dataset_iterator.string_handle()

    # Create elements from iterator
    training_op_1, training_op_2, loss_op_1, loss_op_2 = nn_model(features=features_p, dont_care=dont_care_p,
                                                                  labels=labels_p)

    # Create Config Proto
    config_proto = tf.ConfigProto(log_device_placement=True)
    config_proto.gpu_options.allow_growth = True

    # Create Tensorflow Monitored Session
    sess = tf.train.MonitoredTrainingSession(config=config_proto)

    # Get Handles
    features_dataset_handle_obj = sess.run(features_dataset_handle)
    dont_care_dataset_handle_obj = sess.run(dont_care_dataset_handle)
    label_dataset_handle_obj = sess.run(label_dataset_handle)
    # test_handle = sess.run(te_handle)

    # Epoch For Loop
    for epoch in range(EPOCH):
        batch_algo = (BATCH_SIZE)
        total_batches = int(mnist.train.num_examples / batch_algo)
        # Initialize Training Iterator
        sess.run(features_dataset_iterator.initializer, feed_dict={bs_placeholder: batch_algo})
        # sess.run(dont_care_dataset_iterator.initializer, feed_dict={bs_placeholder: batch_algo})
        sess.run(label_dataset_iterator.initializer, feed_dict={bs_placeholder: batch_algo})
        # sess.run(features_dataset_handle.initializer, feed_dict={bs_placeholder: batch_algo})
        avg_cost = 0.0
        # Loop over all batches
        count = 0
        try:
            # Batch For Loop
            while True:
                _, c = sess.run([training_op_1, loss_op_1],
                                feed_dict={feedable_feature_dataset_handle: features_dataset_handle_obj,
                                           # feedable_dont_care_dataset_handle: dont_care_dataset_handle_obj,
                                           feedable_label_dataset_handle: label_dataset_handle_obj})
                avg_cost += c / total_batches
                count += 1
                # print("Batch:", '%04d' % (count), "cost={:.9f}".format(c))
        except tf.errors.OutOfRangeError:
            if epoch % DISPLAY_STEP == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # avg_cost = 0.0
    # try:
    #     sess.run(test_iterator.initializer)
    #     while True:
    #         c = sess.run(loss_op, feed_dict={handle: test_handle})
    #         avg_cost += c / int(mnist.test.num_examples / mnist.test.num_examples)
    # except tf.errors.OutOfRangeError:
    #     print("Test :", "cost={:.9f}".format(avg_cost))
    print('Total Time Elapsed: {} secs'.format(time.time() - start))

    sess.close()


if __name__ == '__main__':
    main()
