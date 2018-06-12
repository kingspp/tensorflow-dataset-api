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
import json

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

    # Global Variables
    EPOCH = 100
    BATCH_SIZE = 32
    DISPLAY_STEP = 1

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Create Default Placeholder for Features and Labels
    features_p = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='features')
    labels_p = tf.placeholder(dtype=tf.float64, shape=[None, 10], name='labels')

    # Deeplearning Model
    def nn_model(features, labels):
        bn = tf.layers.batch_normalization(features)
        fc1 = tf.layers.dense(bn, 50)
        fc2 = tf.layers.dense(fc1, 50)
        fc2 = tf.layers.dropout(fc2)
        fc3 = tf.layers.dense(fc2, 10)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=fc3))
        op1 = tf.train.AdamOptimizer(learning_rate=0.01)
        optimizer = op1.minimize(loss)
        return optimizer, loss

    nn_model(features=features_p, labels=labels_p)

    init_all_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Initializes all the variables.
        sess.run(init_all_op)
        # Runs to logit.
    graph_def = tf.get_default_graph().as_graph_def()
    # meta_graph_def = tf.train.export_meta_graph()
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

    tf.import_graph_def(graph_def, input_map={'features:0': features, 'labels:0': labels}, name='')

    # tf.train.import_meta_graph(meta_graph_or_file=meta_graph_def,
    #                            input_map={'features:0': features, 'labels:0': labels})

    # Create Handles
    tr_handle = training_iterator.string_handle()
    te_handle = test_iterator.string_handle()

    # Create Config Proto
    config_proto = tf.ConfigProto(log_device_placement=False)
    config_proto.gpu_options.allow_growth = True
    start = time.time()
    # Create Tensorflow Monitored Session
    sess = tf.Session(config=config_proto)
    sess.run(['init', tf.global_variables_initializer()])

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
    json.dump({'internal_time': time.time() - start}, open('/tmp/time.json', 'w'))


if __name__ == '__main__':
    main()
