# -*- coding: utf-8 -*-
"""
| **@created on:** 06/06/18,
| **@author:** Prathyush SP,
| **@version:** v0.0.1
|
| **Description:**
| Basic Placeholders
| **Sphinx Documentation Status:** Complete
|
..todo::
"""

# Imports
from memory_profiler import profile

# @profile(precision=4)
def main():
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data
    import time
    from tensorflow.python.client import timeline

    start = time.time()

    # Global Variables
    EPOCH = 10
    BATCH_SIZE = 32
    DISPLAY_STEP = 1

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Create Placeholders

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
    handle = tf.placeholder(tf.string, shape=[])
    training_iterator = train_dataset.make_one_shot_iterator()
    validation_iterator = valid_dataset.make_one_shot_iterator()

    # Create features and labels
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    features, labels = iterator.get_next()

    features_placeholder = tf.placeholder_with_default(features, [None, mnist.train.images.shape[-1]])
    labels_placeholder = tf.placeholder_with_default(labels, [None, mnist.train.labels.shape[-1]])


    # Deeplearning Model
    def nn_model(features, labels):
        bn = tf.layers.batch_normalization(features)
        fc1 = tf.layers.dense(bn, 50)
        fc2 = tf.layers.dense(fc1, 50)
        fc2 = tf.layers.dropout(fc2)
        fc3 = tf.layers.dense(fc2, 10)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fc3))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
        return optimizer, loss


    # Create elements from iterator
    training_op, loss_op = nn_model(features=features_placeholder, labels=labels_placeholder)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

    # Training without Feed Dict
    with tf.train.MonitoredTrainingSession() as sess:
        sess.run(init_op)
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

        batch_id, epoch_id, total_batches, avg_cost = 0, 0, int(mnist.train.num_examples / BATCH_SIZE), 0
        while not sess.should_stop():
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
            batch_id += 1
        print("Optimization Finished!")

    print('Total Time Elapsed: {} secs'.format(time.time() - start))

    # # Training with Feed Dict
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #     total_batches = int(mnist.train.num_examples / BATCH_SIZE)
    #     for epoch in range(EPOCH):
    #         avg_cost = 0.0
    #         # Loop over all batches
    #         for i in range(total_batches):
    #             batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
    #             _, c = sess.run([training_op, loss_op], feed_dict={features_placeholder: batch_x,
    #                                                                labels_placeholder: batch_y},
    #                             options=options, run_metadata=run_metadata)
    #             # Compute average loss
    #             avg_cost += c / total_batches
    #         # Display logs per epoch step
    #         if epoch % DISPLAY_STEP == 0:
    #             print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
    #     print("Optimization Finished!")
    # print('Total Time Elapsed: {} secs'.format(time.time() - start))

    # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    # chrome_trace = fetched_timeline.generate_chrome_trace_format()
    # with open('timeline_generator-feed_dict.json', 'w') as f:
        # f.write(chrome_trace)

if __name__ == '__main__':
    main()
