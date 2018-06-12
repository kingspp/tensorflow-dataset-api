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
import json
import tensorflow as tf
from benchmark_scripts.common_config import nn_model, config_proto, EPOCH, BATCH_SIZE, DISPLAY_STEP, get_butil, mnist
import time

butil = get_butil(__file__)


@butil.monitor
def main():
    # Create Placeholders
    features_placeholder = tf.placeholder(mnist.train.images.dtype, [None, mnist.train.images.shape[-1]])
    labels_placeholder = tf.placeholder(mnist.train.labels.dtype, [None, mnist.train.labels.shape[-1]])

    # Create elements from iterator
    training_op, loss_op = nn_model(features=features_placeholder, labels=labels_placeholder)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    start = time.time()
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
    json.dump({'internal_time': time.time() - start}, open('/tmp/time.json', 'w'))


if __name__ == '__main__':
    main()
