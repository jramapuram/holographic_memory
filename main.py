import math
import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


from hm import HolographicMemory
from mnist_number import MNIST_Number, full_mnist

#### Config ####
num_models = 100
batch_size = 1
input_size = 784
SEED = 1227
################

def save_fig(m, name):
    plt.figure()
    print 'm shp = ', m.shape
    plt.imshow(m.reshape(28, 28))
    plt.savefig(name, bbox_inches='tight')

def normalize(x, eps=1e-9):
    return (x - np.mean(x)) / (np.abs(x) + 1e-9)

# create a tf session and the holographic memory object
with tf.device("/gpu:0"):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=False, #True,
                                          gpu_options=gpu_options)) as sess:

        memory = HolographicMemory(sess, input_size, num_models, seed=SEED)

        # There are [num_models x num_features] keys
        # keys = memory.split_to_complex(tf.random_normal([num_models, input_size],
        #                                                 seed=SEED*17))
        # print 'past key setup'
        # normalization = memory.complex_mod(keys)
        # keys = tf.Variable(tf.div(keys, tf.complex(normalization, 0.)),
        #                    trainable=False, name="keys")
        # normalization_post = memory.complex_mod(keys)
        keys = [tf.Variable(tf.random_normal([1, input_size], seed=SEED*17+2*i),
                            trainable=False) for i in range(num_models)]
        #keys = [k / tf.nn.l2_normalize(k, 1) for k in keys]

        # Generate some random values & save a test sample
        zero = MNIST_Number(0, full_mnist)
        minibatch, labels = zero.get_batch_iter(batch_size)
        value = tf.constant(minibatch, dtype=tf.float32)
        # value_orig = tf.random_normal([batch_size, input_size], seed=SEED)
        #value = memory.split_to_complex(value_orig)

        sess.run(tf.initialize_all_variables())

        # print the original data
        #value_host, nval, npval = sess.run([value_orig, normalization, normalization_post])
        # value_host = sess.run(value)
        # print 'values to encode[%s] = ' % str(value_host.shape), value_host
        # save_fig(value_host[0], "original_sample.png")
        print 'values to encode : ', str(minibatch.shape)
        save_fig(minibatch[0], "original_sample.png")

        # encode value with the keys
        memories = memory.encode(value, keys)
        memories_host = sess.run(memories)
        print 'encoded memories shape = %s' \
            % (str(memories_host.shape))

        # recover value
        values_recovered = tf.reduce_mean(memory.decode(memories, keys), 0)
        values_recovered_host = sess.run(values_recovered)
        print 'recovered value [%s] = %s' % (values_recovered_host.shape, values_recovered_host)

        #save_fig(normalize(values_recovered_host), "recovered.png")
        save_fig(values_recovered_host, "recovered.png")
