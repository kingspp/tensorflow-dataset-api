# Imports
import os
import sys

if len(sys.argv) <= 1:
    sys.argv.append('cpu')
USE_GPU = True if sys.argv[1] == 'gpu' else False
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if USE_GPU else ""

from benchmark.benchmark import BenchmarkUtil
from benchmark.system_monitors import CPUMonitor, MemoryMonitor, GPUMonitor

butil = BenchmarkUtil(model_name='EP12 Dataset Inbuilt Epoch {}'.format(sys.argv[1]), stats_save_path='/tmp/stats/',
                      monitors=[CPUMonitor, MemoryMonitor, GPUMonitor])


@butil.monitor
def main():
    # Imports
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data
    import time

    start = time.time()
    # Global Variables
    EPOCH = 100
    BATCH_SIZE = 32

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    NUM_SAMPLES = mnist.train.num_examples

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
    while True:
        try:
            _, c = sess.run([training_op, loss_op])
        except tf.errors.OutOfRangeError:
            break
        avg_cost += c / total_batches
        if batch_id == total_batches:
            print("Epoch:", '%04d' % (epoch_id + 1), "cost={:.9f}".format(avg_cost))
            batch_id, avg_cost, cost = 0, 0, []
            epoch_id += 1

        batch_id += 1
    print("Optimization Finished!")

    print('Total Time Elapsed: {} secs'.format(time.time() - start))


if __name__ == '__main__':
    main()
