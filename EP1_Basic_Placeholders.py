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
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from benchmark.benchmark import BenchmarkUtil
from benchmark.system_monitors import CPUMonitor, MemoryMonitor, GPUMonitor

butil = BenchmarkUtil(model_name='Basic Placeholder', stats_save_path='/tmp/stats/',
                      monitors=[CPUMonitor, MemoryMonitor, GPUMonitor])


@butil.monitor
def main():
    # Imports
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data
    import time

    start = time.time()

    # Global Variables
    EPOCH = 1
    BATCH_SIZE = 32
    DISPLAY_STEP = 1

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Create Placeholders
    features_placeholder = tf.placeholder(mnist.train.images.dtype, [None, mnist.train.images.shape[-1]])
    labels_placeholder = tf.placeholder(mnist.train.labels.dtype, [None, mnist.train.labels.shape[-1]])

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

    # Create elements from iterator
    training_op, loss_op = nn_model(features=features_placeholder, labels=labels_placeholder)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config_proto = tf.ConfigProto(log_device_placement=True)
    config_proto.gpu_options.allow_growth = True

    with tf.Session(config=config_proto) as sess:
        sess.run(init_op)
        total_batches = int(mnist.train.num_examples / BATCH_SIZE)
        for epoch in range(EPOCH):
            avg_cost = 0.0
            # Loop over all batches
            for i in range(total_batches):
                batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
                _, c = sess.run([training_op, loss_op], feed_dict={features_placeholder: batch_x,
                                                                   labels_placeholder: batch_y})
                # Compute average loss
                avg_cost += c / total_batches
            # Display logs per epoch step
            if epoch % DISPLAY_STEP == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
        print("Optimization Finished!")

        total_batches, avg_cost = int(mnist.test.num_examples / BATCH_SIZE), 0.0
        for i in range(total_batches):
            batch_x, batch_y = mnist.test.next_batch(BATCH_SIZE)
            c = sess.run(loss_op, feed_dict={features_placeholder: batch_x,
                                             labels_placeholder: batch_y})
            avg_cost += c / total_batches
        print("Validation :", "cost={:.9f}".format(avg_cost))

    print('Total Time Elapsed: {} secs'.format(time.time() - start))


if __name__ == '__main__':
    main()
