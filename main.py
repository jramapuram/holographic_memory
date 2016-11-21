import math
import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from hm import HolographicMemory
from mnist_number import MNIST_Number, full_mnist
from utils import one_hot
from sklearn.preprocessing import normalize

#### Config ####
num_models = 1
batch_size = 1000
input_size = 784
SEED = 1227
keytype = 'std'
################

def save_fig(m, name):
    plt.figure()
    plt.imshow(m.reshape(28, 28))
    plt.savefig(name, bbox_inches='tight')

# def normalize(x, eps=1e-9):
#     return (x - np.mean(x)) / (np.abs(x) + 1e-9)

def gen_unif_keys(input_size, batch_size, num_models):
    assert input_size % 2 == 0
    keys = [tf.Variable(tf.random_uniform([1, input_size],
                                          seed=SEED*17+2*i), #XXX
                        trainable=False) for i in range(num_models)]
    return HolographicMemory.normalize_real_by_complex_abs(keys)

def gen_std_keys(input_size, batch_size, num_models):
    assert input_size % 2 == 0
    keys = [tf.Variable(tf.random_normal([1, input_size],
                                         seed=SEED*17+2*i, #XXX
                                         stddev=1.0/batch_size),
                        trainable=False) for i in range(num_models)]
    return HolographicMemory.normalize_real_by_complex_abs(keys)


def gen_onehot_keys(input_size, batch_size, num_models):
    keys = [tf.Variable(tf.constant(one_hot(input_size, [i]), dtype=tf.float32),
                        trainable=False) for i in range(num_models)]
    #return HolographicMemory.normalize_real_by_complex_abs(keys)
    return keys

def generate_keys(keytype, input_size, batch_size, num_models):
    if keytype == 'onehot':
        keys = gen_onehot_keys(input_size, batch_size, num_models)
    elif keytype == 'normal' or 'std':
        keys = gen_std_keys(input_size, batch_size, num_models)
    elif keytype == 'unif':
        keys = gen_unif_keys(input_size, batch_size, num_models)
    else:
        raise Exception("undefined key type")

    return keys


# create a tf session and the holographic memory object
with tf.device("/gpu:0"):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                          gpu_options=gpu_options)) as sess:
        # initialize our holographic memory
        memory = HolographicMemory(sess, input_size, num_models, seed=SEED)

        # There are num_models x [1 x num_features] keys
        keys = generate_keys(keytype, input_size, batch_size, num_models)

        # Generate some random values & save a test sample
        zero = MNIST_Number(0, full_mnist)
        minibatch, labels = zero.get_batch_iter(batch_size)
        #minibatch, labels = full_mnist.train.next_batch(batch_size)
        value = tf.constant(minibatch, dtype=tf.float32)

        sess.run(tf.initialize_all_variables())
        if keytype != 'onehot':
            memory.verify_key_mod(keys)

        # Get some info on the original data
        print 'values to encode : ', str(minibatch.shape)
        save_fig(minibatch[0], "original_sample.png")

        # encode value with the keys
        # keys = [v + tf.random_normal(value.get_shape().as_list())
        #         for v in tf.split(0, minibatch.shape[0], value)]
        # keys = HolographicMemory.normalize_real_by_complex_abs(keys)
        memories = memory.encode(value, keys)
        memories_host = sess.run(memories)
        print 'encoded memories shape = %s' \
            % (str(memories_host.shape))

        # recover value
        values_recovered = memory.decode(memories, keys)
        values_recovered_host = sess.run(values_recovered)
        print 'recovered value [%s] = %s' % (values_recovered_host.shape, normalize(values_recovered_host))
        #print 'recovered value shape = ', values_recovered_host.shape

        save_fig(normalize(values_recovered_host), "recovered_normalized.png")
        save_fig(values_recovered_host, "recovered.png")
