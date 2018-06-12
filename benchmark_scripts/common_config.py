# -*- coding: utf-8 -*-
"""
| **@created on:** 12/06/18,
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

__all__ = ['nn_model', 'config_proto', 'EPOCH', 'BATCH_SIZE', 'DISPLAY_STEP', 'mnist', 'get_butil']

import sys
from benchmark.benchmark import BenchmarkUtil
from benchmark.system_monitors import CPUMonitor, MemoryMonitor, GPUMonitor
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Global Variables

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


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


config_proto = tf.ConfigProto(log_device_placement=True)
config_proto.gpu_options.allow_growth = True

EPOCH = 100
BATCH_SIZE = 32
DISPLAY_STEP = 1


def get_butil(name):
    if len(sys.argv) <= 1:
        sys.argv.append('cpu')
    USE_GPU = True if sys.argv[1] == 'gpu' else False
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if USE_GPU else ""

    return BenchmarkUtil(model_name=name.split('/')[-1].split('.')[0] + ' {}'.format(sys.argv[1]),
                         stats_save_path='/tmp/stats/',
                         monitors=[CPUMonitor, MemoryMonitor, GPUMonitor])
