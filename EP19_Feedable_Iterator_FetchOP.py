# -*- coding: utf-8 -*-
"""
| **@created on:** 06/06/18,
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
import json
import time

if len(sys.argv) <= 1:
    sys.argv.append('cpu')
USE_GPU = True if sys.argv[1] == 'gpu' else False
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if USE_GPU else ""

from benchmark.benchmark import BenchmarkUtil
from benchmark.system_monitors import CPUMonitor, MemoryMonitor, GPUMonitor

butil = BenchmarkUtil(model_name='EP19 Feedable Iterator Fetch OP {}'.format(sys.argv[1]),
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

    # Create Train Dataset
    train_features_dataset = tf.data.Dataset.from_tensor_slices(mnist.train.images)
    train_label_dataset = tf.data.Dataset.from_tensor_slices(mnist.train.labels)
    train_dataset = tf.data.Dataset.zip((train_features_dataset, train_label_dataset)).repeat(EPOCH).batch(BATCH_SIZE)

    # Create Valid Dataset
    valid_features_dataset = tf.data.Dataset.from_tensor_slices(mnist.test.images)
    valid_label_dataset = tf.data.Dataset.from_tensor_slices(mnist.test.labels)
    valid_dataset = tf.data.Dataset.zip((valid_features_dataset, valid_label_dataset)).batch(
        batch_size=mnist.train.num_examples)

    # Create Dataset Iterator
    handle = tf.placeholder(tf.string, shape=[])
    training_iterator = train_dataset.make_one_shot_iterator()
    validation_iterator = valid_dataset.make_one_shot_iterator()
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)

    # Create features and labels
    features, labels = iterator.get_next()

    # Deeplearning Model
    def nn_model(features, labels):
        bn = tf.layers.batch_normalization(features)
        fc1 = tf.layers.dense(bn, 50)
        fc2 = tf.layers.dense(fc1, 50)
        fc2 = tf.layers.dropout(fc2)
        fc3 = tf.layers.dense(fc2, 10)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=fc3))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
        return optimizer, loss

    tr_handle = training_iterator.string_handle()
    vr_handle = validation_iterator.string_handle()

    # Create elements from iterator
    training_op, loss_op = nn_model(features=features, labels=labels)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config_proto = tf.ConfigProto(log_device_placement=True)
    config_proto.gpu_options.allow_growth = True
    start = time.time()
    with tf.train.MonitoredTrainingSession(config=config_proto) as sess:
        sess.run(init_op)

        # Create Handles
        training_handle = sess.run(tr_handle)
        validation_handle = sess.run(vr_handle)

        batch_id, epoch_id, total_batches, avg_cost = 0, 0, int(mnist.train.num_examples / BATCH_SIZE), 0
        while True:
            try:
                _, c, f, l = sess.run([training_op, loss_op, features, labels], feed_dict={handle: training_handle})
                avg_cost += c / total_batches
                if batch_id == total_batches:
                    if epoch_id % DISPLAY_STEP == 0:
                        print("Epoch:", '%04d' % (epoch_id + 1), "cost={:.9f}".format(avg_cost))
                    batch_id, avg_cost, cost = 0, 0, []
                    epoch_id += 1
                batch_id += 1
            except tf.errors.OutOfRangeError:
                break
        print("Optimization Finished!")
        while True:
            try:
                c = sess.run(loss_op, feed_dict={handle: validation_handle})
                avg_cost += c / total_batches
            except tf.errors.OutOfRangeError:
                break
        print("Validation :", "cost={:.9f}".format(avg_cost))

    print('Total Time Elapsed: {} secs'.format(time.time() - start))
    json.dump({'internal_time': time.time() - start}, open('/tmp/time.json', 'w'))


if __name__ == '__main__':
    main()
