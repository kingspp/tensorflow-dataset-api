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
import json
import tensorflow as tf
from benchmark_scripts.common_config import nn_model, config_proto, EPOCH, BATCH_SIZE, DISPLAY_STEP, get_butil, mnist
import time

butil = get_butil(__file__)


@butil.monitor
def main():
    # Create Default Placeholder for Features and Labels
    features_p = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='features')
    labels_p = tf.placeholder(dtype=tf.float64, shape=[None, 10], name='labels')

    nn_model(features=features_p, labels=labels_p)

    # init_all_op = tf.global_variables_initializer()

    # with tf.Session() as sess:
    #     # Initializes all the variables.
    #     sess.run(init_all_op)
    # Runs to logit.
    # graph_def = tf.get_default_graph().as_graph_def()
    meta_graph_def = tf.train.export_meta_graph()  # graph_def=tf.get_default_graph().as_graph_def(),clear_extraneous_savers=True)
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

    # tf.import_graph_def(graph_def, input_map={'features:0': features, 'labels:0': labels}, name='')

    tf.train.import_meta_graph(meta_graph_or_file=meta_graph_def,
                               input_map={'features:0': features, 'labels:0': labels})

    # Create Handles
    tr_handle = training_iterator.string_handle()
    te_handle = test_iterator.string_handle()

    start = time.time()
    # Create Tensorflow Monitored Session
    sess = tf.train.MonitoredTrainingSession(config=config_proto)
    # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

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
                _, c = sess.run(['Adam', 'Sum:0'], feed_dict={handle: training_handle},
                                # options=options,
                                # run_metadata=run_metadata
                                )
                avg_cost += c / total_batches
                count += 1
                # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                # with open('/tmp/timeline.json', 'w') as f:
                #     f.write(chrome_trace)
                # exit()
                # print("Batch:", '%04d' % (count), "cost={:.9f}".format(c))
        except tf.errors.OutOfRangeError:
            if epoch % DISPLAY_STEP == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    avg_cost = 0.0
    try:
        sess.run(test_iterator.initializer)
        while True:
            c = sess.run('Sum:0', feed_dict={handle: test_handle})
            avg_cost += c / int(mnist.test.num_examples / mnist.test.num_examples)
    except tf.errors.OutOfRangeError:
        print("Test :", "cost={:.9f}".format(avg_cost))
    print('Total Time Elapsed: {} secs'.format(time.time() - start))
    sess.close()
    json.dump({'internal_time': time.time() - start}, open('/tmp/time.json', 'w'))


if __name__ == '__main__':
    main()
