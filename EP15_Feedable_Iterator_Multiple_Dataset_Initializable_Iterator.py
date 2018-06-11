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
    handle = tf.placeholder(tf.string, shape=[])

    # Create Train Dataset
    features_dataset = tf.data.Dataset.from_tensor_slices(mnist.train.images)
    label_dataset = tf.data.Dataset.from_tensor_slices(mnist.train.labels)
    train_dataset = tf.data.Dataset.zip((features_dataset, label_dataset)).batch(batch_size=bs_placeholder)

    # Create Valid Dataset 1
    valid_features_dataset_1 = tf.data.Dataset.from_tensor_slices(mnist.test.images)
    valid_label_dataset_1 = tf.data.Dataset.from_tensor_slices(mnist.test.labels)
    valid_dataset_1 = tf.data.Dataset.zip((valid_features_dataset_1, valid_label_dataset_1)).batch(
        batch_size=mnist.test.num_examples)

    # Create Valid Dataset 2
    valid_features_dataset_2 = tf.data.Dataset.from_tensor_slices(mnist.test.images)
    valid_label_dataset_2 = tf.data.Dataset.from_tensor_slices(mnist.test.labels)
    valid_dataset_2 = tf.data.Dataset.zip((valid_features_dataset_2, valid_label_dataset_2)).batch(
        batch_size=mnist.test.num_examples)

    # Create Valid Dataset 3
    valid_features_dataset_3 = tf.data.Dataset.from_tensor_slices(mnist.test.images)
    valid_label_dataset_3 = tf.data.Dataset.from_tensor_slices(mnist.test.labels)
    valid_dataset_3 = tf.data.Dataset.zip((valid_features_dataset_3, valid_label_dataset_3)).batch(
        batch_size=mnist.test.num_examples)

    # Create Valid Dataset 4
    valid_features_dataset_4 = tf.data.Dataset.from_tensor_slices(mnist.test.images)
    valid_label_dataset_4 = tf.data.Dataset.from_tensor_slices(mnist.test.labels)
    valid_dataset_4 = tf.data.Dataset.zip((valid_features_dataset_4, valid_label_dataset_4)).batch(
        batch_size=mnist.test.num_examples)

    # Create Test Dataset
    test_features_dataset = tf.data.Dataset.from_tensor_slices(mnist.test.images)
    test_label_dataset = tf.data.Dataset.from_tensor_slices(mnist.test.labels)
    test_dataset = tf.data.Dataset.zip((test_features_dataset, test_label_dataset)).batch(
        batch_size=mnist.test.num_examples)

    # Create Dataset Iterators
    training_iterator = train_dataset.make_initializable_iterator()

    validation_iterators = {
        'validation_iterator_1': valid_dataset_1.make_initializable_iterator(),
        'validation_iterator_2': valid_dataset_2.make_initializable_iterator(),
        'validation_iterator_3': valid_dataset_3.make_initializable_iterator(),
        'validation_iterator_4': valid_dataset_4.make_initializable_iterator()
    }

    test_iterator = test_dataset.make_initializable_iterator()

    # Create Feedable Iterator
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, output_shapes=((None, 784), (None, 10)))

    # Create features and labels
    features, labels = iterator.get_next()

    # Create Default Placeholder for Features and Labels
    features_p, labels_p = tf.placeholder_with_default(features, shape=[None, 784]), tf.placeholder_with_default(labels,
                                                                                                                 shape=[
                                                                                                                     None,
                                                                                                                     10])

    # Deeplearning Model
    def nn_model(features, labels):
        bn = tf.layers.batch_normalization(features)
        fc1 = tf.layers.dense(bn, 50)
        fc2 = tf.layers.dense(fc1, 50)
        fc2 = tf.layers.dropout(fc2)
        fc3 = tf.layers.dense(fc2, 10)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=fc3))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
        return optimizer, loss

    # Create Handles
    tr_handle = training_iterator.string_handle()
    vr_handle = {
        'vr_handle_1': validation_iterators['validation_iterator_1'].string_handle(),
        'vr_handle_2': validation_iterators['validation_iterator_2'].string_handle(),
        'vr_handle_3': validation_iterators['validation_iterator_3'].string_handle(),
        'vr_handle_4': validation_iterators['validation_iterator_4'].string_handle()
    }
    te_handle = test_iterator.string_handle()

    # Create elements from iterator
    training_op, loss_op = nn_model(features=features_p, labels=labels_p)

    # Create Config Proto
    config_proto = tf.ConfigProto(log_device_placement=True)
    config_proto.gpu_options.allow_growth = True

    # Create Tensorflow Monitored Session
    sess = tf.train.MonitoredTrainingSession(config=config_proto)

    # Get Handles
    training_handle = sess.run(tr_handle)
    valid_handle = {
        'validation_handle_1': sess.run(vr_handle['vr_handle_1']),
        'validation_handle_2': sess.run(vr_handle['vr_handle_2']),
        'validation_handle_3': sess.run(vr_handle['vr_handle_3']),
        'validation_handle_4': sess.run(vr_handle['vr_handle_4'])
    }
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
                _, c = sess.run([training_op, loss_op], feed_dict={handle: training_handle})
                avg_cost += c / total_batches
                count += 1
                # print("Batch:", '%04d' % (count), "cost={:.9f}".format(c))
        except tf.errors.OutOfRangeError:
            if epoch % DISPLAY_STEP == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
                avg_cost = 0.0
                r = random.randint(0, 3)
                try:
                    sess.run(list(validation_iterators.values())[r].initializer)
                    while True:
                        c = sess.run(loss_op, feed_dict={handle: list(valid_handle.values())[r]})
                        avg_cost += c / int(mnist.test.num_examples / mnist.test.num_examples)
                except tf.errors.OutOfRangeError:
                    print("{} :".format(list(valid_handle.keys())[r]), "cost={:.9f}".format(avg_cost))

    print("Optimization Finished!")

    avg_cost = 0.0
    try:
        sess.run(test_iterator.initializer)
        while True:
            c = sess.run(loss_op, feed_dict={handle: test_handle})
            avg_cost += c / int(mnist.test.num_examples / mnist.test.num_examples)
    except tf.errors.OutOfRangeError:
        print("Test :", "cost={:.9f}".format(avg_cost))
    print('Total Time Elapsed: {} secs'.format(time.time() - start))

    sess.close()


if __name__ == '__main__':
    main()
