# Imports
import time

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from benchmark.benchmark import BenchmarkUtil
from benchmark.system_monitors import CPUMonitor, MemoryMonitor, GPUMonitor

butil = BenchmarkUtil(model_name='Reinitializable Iterator', stats_save_path='/tmp/stats/',
                      monitors=[CPUMonitor, MemoryMonitor, GPUMonitor])


@butil.monitor
def main():
    start = time.time()
    # Global Variables
    EPOCH = 2
    BATCH_SIZE = 32

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    NUM_SAMPLES = mnist.train.num_examples

    # Create Dataset
    features_dataset = tf.data.Dataset.from_tensor_slices(mnist.train.images)
    label_dataset = tf.data.Dataset.from_tensor_slices(mnist.train.labels)
    dataset = tf.data.Dataset.zip((features_dataset, label_dataset)).batch(BATCH_SIZE)

    # Create Dataset Iterator
    iterator = dataset.make_initializable_iterator()

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

    # Create elements from iterator
    training_op, loss_op = nn_model(features=features, labels=labels)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config_proto = tf.ConfigProto(log_device_placement=True)
    config_proto.gpu_options.allow_growth = True

    sess = tf.train.MonitoredTrainingSession(config=config_proto)
    sess.run(init_op)
    batch_id, epoch_id, total_batches, avg_cost = 0, 0, int(NUM_SAMPLES / BATCH_SIZE), 0

    for epoch in range(EPOCH):
        sess.run(iterator.initializer)
        avg_cost = 0.0
        # Loop over all batches
        for i in range(total_batches):
            _, c = sess.run([training_op, loss_op])
            avg_cost += c / total_batches
        print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))

    print("Optimization Finished!")

    print('Total Time Elapsed: {} secs'.format(time.time() - start))


if __name__ == '__main__':
    main()
