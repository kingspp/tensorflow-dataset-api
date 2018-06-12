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
import json

if len(sys.argv) <= 1:
    sys.argv.append('cpu')
USE_GPU = True if sys.argv[1] == 'gpu' else False
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if USE_GPU else ""

from benchmark.benchmark import BenchmarkUtil
from benchmark.system_monitors import CPUMonitor, MemoryMonitor, GPUMonitor

butil = BenchmarkUtil(model_name='EP14 Feedable Iterator Multiple Dataset {}'.format(sys.argv[1]),
                      stats_save_path='/tmp/stats/',
                      monitors=[CPUMonitor, MemoryMonitor, GPUMonitor])


@butil.monitor
def main():
    # Imports
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data
    import time

    # Global Variables
    EPOCH = 100
    BATCH_SIZE = 32
    DISPLAY_STEP = 1

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Create Train Dataset
    train_features_dataset = tf.data.Dataset.from_tensor_slices(mnist.train.images)
    train_label_dataset = tf.data.Dataset.from_tensor_slices(mnist.train.labels)
    train_dataset = tf.data.Dataset.zip((train_features_dataset, train_label_dataset)).repeat(EPOCH).batch(BATCH_SIZE)

    # Create Valid Dataset 1
    valid_features_dataset_1 = tf.data.Dataset.from_tensor_slices(mnist.test.images)
    valid_label_dataset_1 = tf.data.Dataset.from_tensor_slices(mnist.test.labels)
    valid_dataset_1 = tf.data.Dataset.zip((valid_features_dataset_1, valid_label_dataset_1)).batch(
        batch_size=mnist.train.num_examples)

    # Create Valid Dataset 2
    valid_features_dataset_2 = tf.data.Dataset.from_tensor_slices(mnist.test.images)
    valid_label_dataset_2 = tf.data.Dataset.from_tensor_slices(mnist.test.labels)
    valid_dataset_2 = tf.data.Dataset.zip((valid_features_dataset_2, valid_label_dataset_2)).batch(
        batch_size=mnist.train.num_examples)

    # Create Valid Dataset 3
    valid_features_dataset_3 = tf.data.Dataset.from_tensor_slices(mnist.test.images)
    valid_label_dataset_3 = tf.data.Dataset.from_tensor_slices(mnist.test.labels)
    valid_dataset_3 = tf.data.Dataset.zip((valid_features_dataset_3, valid_label_dataset_3)).batch(
        batch_size=mnist.train.num_examples)

    # Create Valid Dataset 4
    valid_features_dataset_4 = tf.data.Dataset.from_tensor_slices(mnist.test.images)
    valid_label_dataset_4 = tf.data.Dataset.from_tensor_slices(mnist.test.labels)
    valid_dataset_4 = tf.data.Dataset.zip((valid_features_dataset_4, valid_label_dataset_4)).batch(
        batch_size=mnist.train.num_examples)

    # Create Valid Dataset 5
    valid_features_dataset_5 = tf.data.Dataset.from_tensor_slices(mnist.test.images)
    valid_label_dataset_5 = tf.data.Dataset.from_tensor_slices(mnist.test.labels)
    valid_dataset_5 = tf.data.Dataset.zip((valid_features_dataset_5, valid_label_dataset_5)).batch(
        batch_size=mnist.train.num_examples)

    # Create Dataset Iterator
    handle = tf.placeholder(tf.string, shape=[])
    training_iterator = train_dataset.make_one_shot_iterator()
    validation_iterator_1 = valid_dataset_1.make_initializable_iterator()
    validation_iterator_2 = valid_dataset_2.make_initializable_iterator()
    validation_iterator_3 = valid_dataset_3.make_initializable_iterator()
    validation_iterator_4 = valid_dataset_4.make_initializable_iterator()
    validation_iterator_5 = valid_dataset_5.make_initializable_iterator()
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
    vr_handle_1 = validation_iterator_1.string_handle()
    vr_handle_2 = validation_iterator_2.string_handle()
    vr_handle_3 = validation_iterator_3.string_handle()
    vr_handle_4 = validation_iterator_4.string_handle()
    vr_handle_5 = validation_iterator_5.string_handle()

    # Create elements from iterator
    training_op, loss_op = nn_model(features=features, labels=labels)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config_proto = tf.ConfigProto(log_device_placement=True)
    config_proto.gpu_options.allow_growth = True
    start = time.time()
    with tf.train.MonitoredTrainingSession(config=config_proto) as sess:
        sess.run(init_op)
        sess.run(validation_iterator_1.initializer)
        sess.run(validation_iterator_2.initializer)
        sess.run(validation_iterator_3.initializer)
        sess.run(validation_iterator_4.initializer)
        sess.run(validation_iterator_5.initializer)

        # Create Handles
        training_handle = sess.run(tr_handle)
        validation_handle_1 = sess.run(vr_handle_1)
        validation_handle_2 = sess.run(vr_handle_2)
        validation_handle_3 = sess.run(vr_handle_3)
        validation_handle_4 = sess.run(vr_handle_4)
        validation_handle_5 = sess.run(vr_handle_5)

        batch_id, epoch_id, total_batches, avg_cost = 0, 0, int(mnist.train.num_examples / BATCH_SIZE), 0
        while True:
            try:
                _, c = sess.run([training_op, loss_op], feed_dict={handle: training_handle})
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

        avg_cost = 0.0
        while True:
            try:
                c = sess.run(loss_op, feed_dict={handle: validation_handle_1})
                avg_cost += c / total_batches
            except tf.errors.OutOfRangeError:
                break
        print("Validation 1 :", "cost={:.9f}".format(avg_cost))

        avg_cost = 0.0
        while True:
            try:
                c = sess.run(loss_op, feed_dict={handle: validation_handle_2})
                avg_cost += c / total_batches
            except tf.errors.OutOfRangeError:
                break
        print("Validation 2 :", "cost={:.9f}".format(avg_cost))

        avg_cost = 0.0
        while True:
            try:
                c = sess.run(loss_op, feed_dict={handle: validation_handle_3})
                avg_cost += c / total_batches
            except tf.errors.OutOfRangeError:
                break
        print("Validation 3 :", "cost={:.9f}".format(avg_cost))

        avg_cost = 0.0
        while True:
            try:
                c = sess.run(loss_op, feed_dict={handle: validation_handle_4})
                avg_cost += c / total_batches
            except tf.errors.OutOfRangeError:
                break
        print("Validation 4 :", "cost={:.9f}".format(avg_cost))

        avg_cost = 0.0
        while True:
            try:
                c = sess.run(loss_op, feed_dict={handle: validation_handle_5})
                avg_cost += c / total_batches
            except tf.errors.OutOfRangeError:
                break
        print("Validation 5 :", "cost={:.9f}".format(avg_cost))

    print('Total Time Elapsed: {} secs'.format(time.time() - start))
    json.dump({'internal_time': time.time() - start}, open('/tmp/time.json', 'w'))


if __name__ == '__main__':
    main()
