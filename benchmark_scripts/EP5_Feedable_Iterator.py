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
import json
import tensorflow as tf
from benchmark_scripts.common_config import nn_model, config_proto, EPOCH, BATCH_SIZE, DISPLAY_STEP, get_butil, mnist
import time

butil = get_butil(__file__)


@butil.monitor
def main():
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

    tr_handle = training_iterator.string_handle()
    vr_handle = validation_iterator.string_handle()

    # Create elements from iterator
    training_op, loss_op = nn_model(features=features, labels=labels)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    start = time.time()
    with tf.train.MonitoredTrainingSession(config=config_proto) as sess:
        # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        sess.run(init_op)

        # Create Handles
        training_handle = sess.run(tr_handle)
        validation_handle = sess.run(vr_handle)

        batch_id, epoch_id, total_batches, avg_cost = 0, 0, int(mnist.train.num_examples / BATCH_SIZE), 0
        while True:
            try:
                _, c = sess.run([training_op, loss_op], feed_dict={handle: training_handle},
                                # options=options,
                                # run_metadata=run_metadata
                                )
                avg_cost += c / total_batches
                if batch_id == total_batches:
                    if epoch_id % DISPLAY_STEP == 0:
                        print("Epoch:", '%04d' % (epoch_id + 1), "cost={:.9f}".format(avg_cost))
                    batch_id, avg_cost, cost = 0, 0, []
                    epoch_id += 1
                    # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    # with open('/tmp/timeline-ep5.json', 'w') as f:
                    #     f.write(chrome_trace)
                    # exit()
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
