import math
import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


from hm import HolographicMemory
from mnist_number import MNIST_Number, full_mnist

#### Config ####
num_models = 1
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
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=False, #True,
                                          gpu_options=gpu_options)) as sess:

        memory = HolographicMemory(sess, input_size, num_models, seed=SEED)

        # There are [num_models x num_features] keys
        keys = [tf.Variable(tf.random_normal([1, input_size], seed=SEED*17+2*i), #XXX
                            trainable=False) for i in range(num_models)]
        #keys = [k / tf.nn.l2_normalize(k, 1) for k in keys]

        # Generate some random values & save a test sample
        zero = MNIST_Number(0, full_mnist)
        minibatch, labels = zero.get_batch_iter(batch_size)
        value = tf.constant(minibatch, dtype=tf.float32)

        sess.run(tf.initialize_all_variables())

        # Get some info on the original data
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
