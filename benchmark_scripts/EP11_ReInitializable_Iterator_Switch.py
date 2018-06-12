# -*- coding: utf-8 -*-
"""
| **@created on:** 08/06/18,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| 
|
| **Sphinx Documentation Status:** --
|
..todo::
"""

import json
import tensorflow as tf
from benchmark_scripts.common_config import nn_model, config_proto, EPOCH, BATCH_SIZE, DISPLAY_STEP, get_butil, mnist
import time

butil = get_butil(__file__)


@butil.monitor
def main():
    # Create Dataset
    train_features_dataset = tf.data.Dataset.from_tensor_slices(mnist.train.images)
    train_label_dataset = tf.data.Dataset.from_tensor_slices(mnist.train.labels)
    train_dataset = tf.data.Dataset.zip((train_features_dataset, train_label_dataset)).repeat(EPOCH).batch(BATCH_SIZE)

    # Create Valid Dataset
    valid_features_dataset = tf.data.Dataset.from_tensor_slices(mnist.test.images)
    valid_label_dataset = tf.data.Dataset.from_tensor_slices(mnist.test.labels)
    valid_dataset = tf.data.Dataset.zip((valid_features_dataset, valid_label_dataset)).batch(
        batch_size=mnist.train.num_examples)

    # Create Dataset Iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)

    # Create features and labels
    features, labels = iterator.get_next()

    # Create Initialization Op
    train_init_op = iterator.make_initializer(train_dataset)
    valid_init_op = iterator.make_initializer(valid_dataset)

    # Create elements from iterator
    training_op, loss_op = nn_model(features=features, labels=labels)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    start = time.time()
    with tf.train.MonitoredTrainingSession(config=config_proto) as sess:
        sess.run(init_op)
        sess.run(train_init_op)
        batch_id, epoch_id, total_batches, avg_cost = 0, 0, int(mnist.train.num_examples / BATCH_SIZE), 0
        while True:
            try:
                _, c = sess.run([training_op, loss_op])
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

        sess.run(valid_init_op)
        while True:
            try:
                c = sess.run(loss_op)
                avg_cost += c / total_batches
            except tf.errors.OutOfRangeError:
                break
        print("Validation :", "cost={:.9f}".format(avg_cost))

    print('Total Time Elapsed: {} secs'.format(time.time() - start))
    json.dump({'internal_time': time.time() - start}, open('/tmp/time.json', 'w'))


if __name__ == '__main__':
    main()
