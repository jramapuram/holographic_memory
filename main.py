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
num_models = 3
batch_size = 2
input_size = 784
SEED = 1337
keytype = 'std'
pseudokeys = True
normalize_keys = False
################

def save_fig(m, name):
    plt.figure()
    plt.imshow(m.reshape(28, 28))
    plt.savefig(name, bbox_inches='tight')

def gen_unif_keys(input_size, batch_size):
    assert input_size % 2 == 0
    keys = [tf.Variable(tf.random_uniform([1, input_size],
                                          seed=SEED*17+2*i if SEED else None), #XXX
                        trainable=False, name="key_%d"%i) for i in range(batch_size)]
    return keys

def gen_std_keys(input_size, batch_size):
    assert input_size % 2 == 0
    keys = [tf.Variable(tf.random_normal([1, input_size],
                                         seed=SEED*17+2*i if SEED else None, #XXX
                                         stddev=1.0/batch_size),
                        trainable=False, name="key_%d"%i) for i in range(batch_size)]
    return keys


def gen_onehot_keys(input_size, batch_size):
    keys = [tf.Variable(tf.constant(one_hot(input_size, [i]), dtype=tf.float32),
                        trainable=False, name="key_%d"%i) for i in range(batch_size)]
    return keys

def generate_keys(keytype, input_size, batch_size):
    if keytype == 'onehot':
        keys = gen_onehot_keys(input_size, batch_size)
    elif keytype == 'normal' or 'std':
        keys = gen_std_keys(input_size, batch_size)
    elif keytype == 'unif':
        keys = gen_unif_keys(input_size, batch_size)
    else:
        raise Exception("undefined key type")

    return keys


# create a tf session and the holographic memory object
with tf.device("/gpu:0"):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                          gpu_options=gpu_options)) as sess:
        # initialize our holographic memory
        memory = HolographicMemory(sess, input_size, batch_size, num_models, seed=SEED)

        # Generate some random values & save a test sample
        # zero = MNIST_Number(0, full_mnist)
        # minibatch, labels = zero.get_batch_iter(batch_size)
        minibatch, labels = full_mnist.train.next_batch(batch_size)
        value = tf.constant(minibatch, dtype=tf.float32)

        # There are num_models x [1 x num_features] keys
        # They are generated by either:
        #     1) Randomly Generated [pseudokeys=True]
        #     2) From a noisy version of the data
        if pseudokeys:
            print 'generating pseudokeys...'
            keys = generate_keys(keytype, input_size, batch_size)
        else:
            print 'utilizing real keys with N(0,I)...'
            keys = [v + tf.random_normal(v.get_shape().as_list())
                    for v in tf.split(0, minibatch.shape[0], value)]

        # Normalize our keys to mod 1
        if normalize_keys:
            keys = HolographicMemory.normalize_real_by_complex_abs(keys)

        sess.run(tf.initialize_all_variables())

        # do a little validation on the keys
        if normalize_keys and keytype != 'onehot':
            memory.verify_key_mod(keys)

        # Get some info on the original data
        print 'values to encode : ', str(minibatch.shape)
        save_fig(minibatch[0], "original_sample.png")

        # encode value with the keys
        memories = memory.encode(value, keys)
        memories_host = sess.run(memories)
        print 'encoded memories shape = %s' \
            % (str(memories_host.shape))
        #print 'em = ', memories_host

        # recover value
        values_recovered = tf.reduce_sum(memory.decode(memories, [keys[0]]), 0)
        values_recovered_host = sess.run(values_recovered)
        #print 'recovered value [%s] = %s' % (values_recovered_host.shape, normalize(values_recovered_host))
        print 'recovered value shape = ', values_recovered_host.shape

        # for mat, i in zip(values_recovered_host, range(len(values_recovered_host))):
        #     save_fig(normalize(mat), "recovered_normalized_%d.png" % i)
        #     save_fig(mat, "recovered_%d.png" % i)

        save_fig(normalize(values_recovered_host), "recovered_normalized.png")
        save_fig(values_recovered_host, "recovered.png")
