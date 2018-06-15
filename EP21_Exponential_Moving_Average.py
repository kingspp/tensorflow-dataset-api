import numpy as np
import tensorflow as tf
import json
# Imports

from tensorflow.examples.tutorials.mnist import input_data
import time


def moving_average_op(loss_op):
    cumulative = tf.Variable(0, dtype=tf.float32)
    divisor = tf.Variable(0, dtype=tf.float32)
    init_cum_vars = tf.variables_initializer(var_list=[cumulative, divisor])
    cumulative = cumulative.assign_add(loss_op)
    divisor = divisor.assign_add(1)
    avg = tf.div(cumulative, divisor)
    return avg, init_cum_vars


start = time.time()

# Global Variables
EPOCH = 100
BATCH_SIZE = 32
DISPLAY_STEP = 1

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Create Dataset
features_dataset = tf.data.Dataset.from_tensor_slices(mnist.train.images)
label_dataset = tf.data.Dataset.from_tensor_slices(mnist.train.labels)
dataset = tf.data.Dataset.zip((features_dataset, label_dataset)).batch(BATCH_SIZE).repeat(EPOCH)

# Create Dataset Iterator
iterator = dataset.make_one_shot_iterator()

# Create features and labels
features, labels = iterator.get_next()


# Deeplearning Model
def nn_model(features, labels):
    bn = tf.layers.batch_normalization(features)
    fc1 = tf.layers.dense(bn, 50)
    fc2 = tf.layers.dense(fc1, 50)
    fc2 = tf.layers.dropout(fc2)
    fc3 = tf.layers.dense(fc2, 10)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fc3))
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    m = ema.apply([loss])
    av = ema.average(loss)
    avg, var_init = moving_average_op(loss)
    avg_ni, var_not_init = moving_average_op(loss)
    with tf.control_dependencies([m]):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    return optimizer, loss, av, avg, var_init, avg_ni


# Create elements from iterator
training_op, loss_op, av, avg, var_init, avg_ni = nn_model(features=features, labels=labels)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# ec, emanual, emac = [], [], []

cost_plots = {
    'Algorithm Based Average': [],
    'Manual Average': [],
    'Exponential Moving Average': [],
    'Tensor Average': [],
    'Tensor Average without Init': []
}

with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    batch_id, epoch_id, total_batches, avg_cost, avg_cost_manual = 0, 0, int(
        mnist.train.num_examples / BATCH_SIZE), 0, []
    while not sess.should_stop():
        _, c, a, tavg, tni = sess.run([training_op, loss_op, av, avg, avg_ni])
        avg_cost += c / total_batches
        avg_cost_manual.append(c)
        if batch_id == total_batches:
            if epoch_id % DISPLAY_STEP == 0:
                print("Epoch:", '%04d' % (epoch_id + 1),
                      "cost={:.9f}, manual={} ema={}, tensor_avg={}, tensor_avg_not_init={}".format(
                          avg_cost, sum(avg_cost_manual) / len(avg_cost_manual), a, tavg, tni))
                sess.run([var_init])
                cost_plots['Algorithm Based Average'].append(float(avg_cost))
                cost_plots['Exponential Moving Average'].append(float(a))
                cost_plots['Manual Average'].append(float(sum(avg_cost_manual) / len(avg_cost_manual)))
                cost_plots['Tensor Average'].append(float(tavg))
                cost_plots['Tensor Average without Init'].append(float(tni))
            batch_id, avg_cost, avg_cost_manual, cost = 0, 0, [], []
            epoch_id += 1
        batch_id += 1
    print("Optimization Finished!")

print('Total Time Elapsed: {} secs'.format(time.time() - start))

json.dump(cost_plots, open('/tmp/cost_vals.json', 'w'))
