# -*- coding: utf-8 -*-
"""
| **@created on:** 11/06/18,
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

butil = BenchmarkUtil(
    model_name='EP15 Feedable Iterator, Multiple Dataset, Initializable Iterator {}'.format(sys.argv[1]),
    stats_save_path='/tmp/stats/',
    monitors=[CPUMonitor, MemoryMonitor, GPUMonitor])


@butil.monitor
def main():
    # Imports
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data

    start = time.time()

    # Global Variables
    EPOCH = 100
    BATCH_SIZE = 32
    DISPLAY_STEP = 1

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Create Default Placeholder for Features and Labels
    features_p = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='features')
    labels_p = tf.placeholder(dtype=tf.float64, shape=[None, 10], name='labels')
    dont_care_p = tf.placeholder(dtype=tf.float64, shape=[None, 784], name='dont_care')

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

    nn_model(features=features_p, labels=labels_p, dont_care=dont_care_p)

    # graph_def = tf.get_default_graph().as_graph_def()
    meta_graph_model = tf.train.export_meta_graph()
    tf.reset_default_graph()

    # Create Dataset Handle
    handle = tf.placeholder(tf.string, shape=[])
    bs_placeholder = tf.placeholder(dtype=tf.int64)

    # Create Train Dataset
    features_dataset = tf.data.Dataset.from_tensor_slices(mnist.train.images)
    label_dataset = tf.data.Dataset.from_tensor_slices(mnist.train.labels)

    train_dataset = tf.data.Dataset.zip((features_dataset, label_dataset)).batch(batch_size=bs_placeholder)

    # Create Test Dataset
    test_features_dataset = tf.data.Dataset.from_tensor_slices(mnist.test.images)
    test_label_dataset = tf.data.Dataset.from_tensor_slices(mnist.test.labels)
    test_dataset = tf.data.Dataset.zip((test_features_dataset, test_label_dataset)).batch(
        batch_size=mnist.test.num_examples)

    # Create Dataset Iterators
    training_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    # Create Feedable Iterator
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, output_shapes=((None, 784), (None, 10)))

    # Create features and labels
    features, labels = iterator.get_next()

    tr_handle = training_iterator.string_handle()
    te_handle = test_iterator.string_handle()

    meta_graph_data = tf.train.export_meta_graph()

    tf.reset_default_graph()

    tf.train.import_meta_graph(meta_graph_or_file=meta_graph_data, clear_devices=True)
    # Create Handles
    tf.train.import_meta_graph(meta_graph_or_file=meta_graph_model, clear_devices=True,
                               input_map={'features:0': tf.get_default_graph().get_tensor_by_name('IteratorGetNext:0'),
                                          'labels:0': tf.get_default_graph().get_tensor_by_name("IteratorGetNext:1")})

    # Create Config Proto
    config_proto = tf.ConfigProto(log_device_placement=True)
    config_proto.gpu_options.allow_growth = True

    # Create Tensorflow Monitored Session
    sess = tf.train.MonitoredTrainingSession(config=config_proto)

    # Get Handles
    training_handle = sess.run(tr_handle)
    test_handle = sess.run(te_handle)

    # Epoch For Loop
    for epoch in range(EPOCH):
        batch_algo = (BATCH_SIZE)
        total_batches = int(mnist.train.num_examples / batch_algo)
        # Initialize Training Iterator
        sess.run(training_iterator.initializer, feed_dict={bs_placeholder: batch_algo})
        avg_cost = 0.0
        # Loop over all batches
        count = 0
        try:
            # Batch For Loop
            while True:
                _, c = sess.run(['Adam', 'Mean:0'], feed_dict={handle: training_handle})
                avg_cost += c / total_batches
                count += 1
                # print("Batch:", '%04d' % (count), "cost={:.9f}".format(c))
        except tf.errors.OutOfRangeError:
            if epoch % DISPLAY_STEP == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    avg_cost = 0.0
    try:
        sess.run(test_iterator.initializer)
        while True:
            c = sess.run('Mean:0', feed_dict={handle: test_handle})
            avg_cost += c / int(mnist.test.num_examples / mnist.test.num_examples)
    except tf.errors.OutOfRangeError:
        print("Test :", "cost={:.9f}".format(avg_cost))
    print('Total Time Elapsed: {} secs'.format(time.time() - start))
    sess.close()

    sess = tf.train.MonitoredTrainingSession(config=config_proto)

    # Get Handles
    training_handle = sess.run(tr_handle)
    test_handle = sess.run(te_handle)

    # Epoch For Loop
    for epoch in range(EPOCH):
        batch_algo = (BATCH_SIZE)
        total_batches = int(mnist.train.num_examples / batch_algo)
        # Initialize Training Iterator
        sess.run(training_iterator.initializer, feed_dict={bs_placeholder: batch_algo})
        avg_cost = 0.0
        # Loop over all batches
        count = 0
        try:
            # Batch For Loop
            while True:
                _, c = sess.run(['Adam', 'Mean:0'], feed_dict={handle: training_handle})
                avg_cost += c / total_batches
                count += 1
                # print("Batch:", '%04d' % (count), "cost={:.9f}".format(c))
        except tf.errors.OutOfRangeError:
            if epoch % DISPLAY_STEP == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    avg_cost = 0.0
    try:
        sess.run(test_iterator.initializer)
        while True:
            c = sess.run('Mean:0', feed_dict={handle: test_handle})
            avg_cost += c / int(mnist.test.num_examples / mnist.test.num_examples)
    except tf.errors.OutOfRangeError:
        print("Test :", "cost={:.9f}".format(avg_cost))
    print('Total Time Elapsed: {} secs'.format(time.time() - start))
    sess.close()


if __name__ == '__main__':
    main()
