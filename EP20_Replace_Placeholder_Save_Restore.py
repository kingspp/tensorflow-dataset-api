# -*- coding: utf-8 -*-
"""
| **@created on:** 13/06/18,
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
import os
import sys
import time
import json
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

if len(sys.argv) <= 1:
    sys.argv.append('cpu')
USE_GPU = True if sys.argv[1] == 'gpu' else False
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if USE_GPU else ""

from benchmark.benchmark import BenchmarkUtil
from benchmark.system_monitors import CPUMonitor, MemoryMonitor, GPUMonitor

butil = BenchmarkUtil(
    model_name='EP17 Replaceable Placeholder {}'.format(sys.argv[1]),
    stats_save_path='/tmp/stats/',
    monitors=[CPUMonitor, MemoryMonitor, GPUMonitor])

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


def create_dataset():
    # Create Dataset Handle
    bs_placeholder = tf.placeholder(dtype=tf.int64)
    handle = tf.placeholder(tf.string, shape=[])
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

    # Create Handles
    tr_handle = training_iterator.string_handle()
    te_handle = test_iterator.string_handle()

    return {'data': [features, labels], 'handles': [tr_handle, te_handle],
            'iterators': [training_iterator, test_iterator], 'placeholders': [bs_placeholder, handle]}


# @butil.monitor
def main():
    # Imports

    # Global Variables
    EPOCH = 1
    BATCH_SIZE = 32
    DISPLAY_STEP = 1

    # Create Default Placeholder for Features and Labels
    features_p = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='features')
    dont_care_p = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='features')
    labels_p = tf.placeholder(dtype=tf.float64, shape=[None, 10], name='labels')

    # Deeplearning Model
    def nn_model(features, dont_care, labels):
        bn = tf.layers.batch_normalization(features)
        fc1 = tf.layers.dense(bn, 50)
        fc2 = tf.layers.dense(fc1, 50)
        fc2 = tf.layers.dropout(fc2)
        fc3 = tf.layers.dense(fc2, 10)
        l1 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=fc3))
        o1 = tf.train.AdamOptimizer(learning_rate=0.01).minimize(l1)
        # Create an op1 graph

        fc4 = tf.layers.dense(dont_care, 10)
        fc5 = tf.concat([fc3, fc4], axis=0)
        l2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=fc5))
        o2 = tf.train.AdamOptimizer(learning_rate=0.01).minimize(l2)
        # Create an o2 graph

        return o1, o2, l1, l2

    nn_model(features=features_p, labels=labels_p, dont_care=dont_care_p)

    # init_all_op = tf.global_variables_initializer()

    # with tf.Session() as sess:
    #     # Initializes all the variables.
    #     sess.run(init_all_op)
    # Runs to logit.
    GLOBAL_MODEL_GRAPH = tf.get_default_graph().as_graph_def()
    # print(tf.get_default_graph().get_tensor_by_name(l1.name))
    OP1_GRAPH = tf.graph_util.extract_sub_graph(GLOBAL_MODEL_GRAPH, ['Adam'])
    OP2_GRAPH = tf.graph_util.extract_sub_graph(GLOBAL_MODEL_GRAPH, ['Adam_1'])
    OP1_OP2_GRAPH = tf.graph_util.extract_sub_graph(GLOBAL_MODEL_GRAPH, ['Adam', 'Adam_1'])


    op1_meta_graph = tf.train.export_meta_graph(graph_def=OP1_GRAPH)
    op2_meta_graph = tf.train.export_meta_graph(graph_def=OP2_GRAPH)
    op1_op2_meta_graph = tf.train.export_meta_graph(graph_def=OP1_OP2_GRAPH)
    # graph_def=tf.get_default_graph().as_graph_def(),clear_extraneous_savers=True)
    tf.reset_default_graph()

    dataset_1 = create_dataset()

    tf.train.import_meta_graph(meta_graph_or_file=op1_meta_graph,
                               input_map={'features:0': dataset_1['data'][0], 'labels:0': dataset_1['data'][1]})

    # Create Config Proto
    config_proto = tf.ConfigProto(log_device_placement=True)
    config_proto.gpu_options.allow_growth = True
    start = time.time()
    # Create Tensorflow Monitored Session
    sess = tf.train.MonitoredTrainingSession(config=config_proto)
    # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

    # Get Handles
    training_handle = sess.run(dataset_1['handles'][0])
    test_handle = sess.run(dataset_1['handles'][1])

    # Epoch For Loop
    for epoch in range(EPOCH):
        batch_algo = (BATCH_SIZE)
        total_batches = int(mnist.train.num_examples / batch_algo)
        # Initialize Training Iterator
        sess.run(dataset_1['iterators'][0].initializer, feed_dict={dataset_1['placeholders'][0]: batch_algo})
        avg_cost = 0.0
        # Loop over all batches
        count = 0
        try:
            # Batch For Loop
            while True:
                _, c = sess.run(['Adam', 'Sum:0'], feed_dict={dataset_1['placeholders'][1]: training_handle},
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
        sess.run(dataset_1['iterators'][1].initializer)
        while True:
            c = sess.run('Sum:0', feed_dict={dataset_1['placeholders'][1]: test_handle})
            avg_cost += c / mnist.test.num_examples
    except tf.errors.OutOfRangeError:
        print("Test :", "cost={:.9f}".format(avg_cost))
    print('Total Time Elapsed: {} secs'.format(time.time() - start))



    sess.close()
    json.dump({'internal_time': time.time() - start}, open('/tmp/time.json', 'w'))


if __name__ == '__main__':
    main()
