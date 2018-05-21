"""
A set of helper functions for tensorflow
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import tensorflow as tf


#gpu_config:
def gpu_config(gpu_id=None):
    if gpu_id is not None and gpu_id != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.allow_soft_placement = True
    else:
        config = tf.ConfigProto()
    return config


