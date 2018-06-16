# -*- coding: utf-8 -*-
"""
| **@created on:** 06/06/18,
| **@author:** Prathyush SP,
| **@version:** v0.0.1
|
| **Description:**
| Feedable Dataset
|
| **Sphinx Documentation Status:** Complete
|
..todo::
"""
import json
import tensorflow as tf
from benchmark_scripts.common_config import nn_model, config_proto, EPOCH, BATCH_SIZE, DISPLAY_STEP, get_butil, mnist
import time
from threading import Thread

butil = get_butil(__file__)

THREADS_ENABLED = True


def saver_fn(i, session: tf.Session, saver: tf.train.Saver):
    print('Saving Session {}. . .'.format(i))
    saver.save(session, '/tmp/new_save', global_step=i)


@butil.monitor
def main():
    # Create Placeholders
    features_placeholder = tf.placeholder(mnist.train.images.dtype, [None, mnist.train.images.shape[-1]])
    labels_placeholder = tf.placeholder(mnist.train.labels.dtype, [None, mnist.train.labels.shape[-1]])

    # Create Tensor slices from placeholders
    train_dataset = tf.data.Dataset.from_tensor_slices(features_placeholder)
    label_dataset = tf.data.Dataset.from_tensor_slices(labels_placeholder)

    # Create Dataset
    dataset = tf.data.Dataset.zip((train_dataset, label_dataset)).batch(BATCH_SIZE).repeat(EPOCH)

    # Create Dataset Iterator
    iterator = dataset.make_initializable_iterator()

    # Create features and labels
    features, labels = iterator.get_next()

    # Create elements from iterator
    training_op, loss_op = nn_model(features=features, labels=labels)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    threads = []
    saver = tf.train.Saver()
    start = time.time()
    with tf.Session() as sess:
        sess.run(init_op)
        batch_id, total_batches, avg_cost = 0, int(mnist.train.num_examples / BATCH_SIZE), 0
        for i in range(EPOCH):
            sess.run(iterator.initializer)
            try:
                while True:
                    _, c = sess.run([training_op, loss_op])
                    avg_cost += c / total_batches
                    batch_id += 1
            except tf.errors.OutOfRangeError:
                if i % DISPLAY_STEP == 0:
                    print("Epoch:", '%04d' % (i + 1), "cost={:.9f}".format(avg_cost))
                    if THREADS_ENABLED:
                        threads.append(Thread(target=saver_fn, args=(i, sess, saver)))
                        threads[-1].start()
                    else:
                        print('Saving Session {}. . .'.format(i))
                        saver.save(sess, '/tmp/new_save', global_step=i)
                batch_id, avg_cost, cost = 0, 0, []

        print("Optimization Finished!")
        if THREADS_ENABLED:
            [i.join() for i in threads]

    print('Total Time Elapsed: {} secs'.format(time.time() - start))
    json.dump({'internal_time': time.time() - start}, open('/tmp/time.json', 'w'))


if __name__ == '__main__':
    main()
