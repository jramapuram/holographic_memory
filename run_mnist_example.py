import os
import math
import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from hm import HolographicMemory
from mnist_number import MNIST_Number, full_mnist
from utils import one_hot
#from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

#################################################################################################
#                             Configuration parameters & Defaults                               #
#################################################################################################
flags = tf.flags
flags.DEFINE_integer("num_copies", 3, "Number of copies to make.")
flags.DEFINE_integer("minibatch_size", 2, "Number of samples to use in minibatch")
flags.DEFINE_integer("batch_size", 64, "Total number of samples")
flags.DEFINE_integer("seed", None, "Fixed seed to get reproducible results.")
flags.DEFINE_string("keytype", "normal", "Use N(0, I) keys")
flags.DEFINE_bool("pseudokeys", 1, "Use synthetically generated keys or [data + error] as keys")
flags.DEFINE_bool("complex_normalize", 0, "Normalize keys via complex mod.")
flags.DEFINE_bool("l2_normalize", 0, "Normalize keys via l2 norm.")
flags.DEFINE_string("device", "/gpu:0", "Compute device.")
flags.DEFINE_boolean("allow_soft_placement", False, "Soft device placement.")
flags.DEFINE_float("device_percentage", 0.9, "Amount of memory to use on device.")
FLAGS = flags.FLAGS
#################################################################################################

def save_fig(m, name):
    plt.figure()
    plt.imshow(m.reshape(28, 28))
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def gen_unif_keys(input_size, batch_size, seed):
    assert input_size % 2 == 0
    np.random.seed(seed*17+1 if seed else None)
    return np.random.random([batch_size, input_size])

def gen_std_keys(input_size, batch_size, seed):
    assert input_size % 2 == 0
    np.random.seed(seed*17+1 if seed else None)
    return np.random.randn(batch_size, input_size)

# Note: This will only work if batch_size <= input_size
def gen_onehot_keys(input_size, batch_size, targets):
    return np.vstack([one_hot(input_size, [t]) for t in targets])

def normalize(x, scale_range=True):
    if len(x.shape) == 2:
        cleaned = (x - np.expand_dims(np.mean(x, axis=1), 1)) / np.expand_dims(np.clip(np.std(x, axis=1), 1e-9, 1e25), 1)
    elif len(x.shape) == 1:
        cleaned = (x - np.mean(x)) / np.clip(np.std(x), 1e-9, 1e25)
    else:
        raise Exception("Unknown shape provided")

    return MinMaxScaler().fit_transform(cleaned) if scale_range else cleaned

def generate_keys(keytype, input_size, batch_size, seed, targets=None):
    if keytype == 'onehot':
        assert targets is not None, "targets need to be provided for one-hot"
        keys = gen_onehot_keys(input_size, batch_size, targets)
    elif keytype == 'normal' or 'std':
        keys = gen_std_keys(input_size, batch_size, seed)
    elif keytype == 'unif':
        keys = gen_unif_keys(input_size, batch_size, seed)
    else:
        raise Exception("undefined key type")

    return keys

def build_model(sess, keys, values, num_copies, batch_size=None):
    input_size = values.get_shape().as_list()[1]
    batch_size = values.get_shape().as_list()[0] if batch_size is None else batch_size

    # initialize our holographic memory
    memory = HolographicMemory(sess, input_size, num_copies,
                               complex_normalize=FLAGS.complex_normalize,
                               l2_normalize=FLAGS.l2_normalize,
                               seed=FLAGS.seed)

    print 'keys = ', keys.get_shape().as_list()
    encoder = memory.encode(values, keys, batch_size=batch_size)
    decoder = memory.decode(values, keys, num_keys=batch_size)

    return memory, encoder, decoder

def encode(sess, memory, encoder, values, keys, full_batch_host, keys_host, batch_size):
    full_batch_size = full_batch_host.shape[0]
    assert full_batch_size >= batch_size, "full batch size needs to be >= mini-batch size"
    memories_host = np.zeros([memory.num_models, memory.input_size])
    print 'full_batch_size = ', full_batch_size, 'minibatch_size = ', batch_size

    for begin,end in zip(range(0, full_batch_size, batch_size),
                         range(batch_size, full_batch_size+1, batch_size)):
        feed_dict={keys: keys_host[begin:end],
                   values: full_batch_host[begin:end]}

        # encode value with the keys
        memories_host += sess.run(encoder, feed_dict=feed_dict)

    #np.savetxt("encoded.csv", memories_host, delimiter=",")
    return memories_host


def decode(sess, memory, decoder, values, keys, memories_host, keys_host, key_batch):
    recovered_host = []
    keys_size = keys_host.shape[0]  # this can be 1 or more keys
    assert keys_size >= key_batch, "key full batch size needs to be >= key mini-batch size"
    print 'key size = ', keys_size, 'key_batch = ', key_batch

    for begin,end in zip(range(0, keys_size, key_batch),
                         range(key_batch, keys_size+1, key_batch)):
        #print 'b/e decode = ', begin, end, ' | key_batch = ', key_batch
        feed_dict={keys: keys_host[begin:end],
                   values: memories_host}

        # decode values using keys and memories
        r = sess.run(decoder, feed_dict=feed_dict)
        recovered_host.append(r)

    recovered = np.vstack(recovered_host)
    print 'values_recovered shape = ', recovered.shape
    return recovered


def main():
    # create a tf session and the holographic memory object
    with tf.device(FLAGS.device):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.device_percentage)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                              gpu_options=gpu_options)) as sess:
            input_size = 784  # MNIST input size [28, 28]

            # create placeholders for our items
            # FLAGS.minibatch_size
            keys = tf.placeholder(tf.float32, [None, input_size], name="keys")
            values = tf.placeholder(tf.float32, [None, input_size], name="values")

            # Generate some random values & save a test sample
            #minibatch, labels = MNIST_Number(0, full_mnist).get_batch_iter(FLAGS.minibatch_size)
            minibatch, labels = full_mnist.train.next_batch(FLAGS.batch_size)

            # Get some info on the original data
            print 'values to encode : ', str(minibatch.shape)
            # for i in range(len(minibatch)):
            #     save_fig(minibatch[i], "imgs/original_%d.png" %i)

            # There are num_copies x [1 x num_features] keys
            # They are generated by either:
            #     1) Randomly Generated [FLAGS.pseudokeys=True]
            #     2) From a noisy version of the data
            if FLAGS.pseudokeys:
                print 'generating pseudokeys...'
                keys_host = generate_keys(FLAGS.keytype, input_size,
                                          FLAGS.batch_size,
                                          FLAGS.seed, targets=labels)
                print keys_host
            else:
                print 'utilizing real data + N(0,I) as keys...'
                # keys = [tf.add(v, tf.random_normal(v.get_shape().as_list(), seed=FLAGS.seed*17+2*i), name="keys_%d"%i)
                #         for v, i in zip(tf.split(0, minibatch.shape[0], value), range(minibatch.shape[0]))]
                np.random.seed(FLAGS.seed*33 if FLAGS.seed else None)
                keys_host = normalize(minibatch + np.random.randn(FLAGS.batch_size, input_size),
                                      scale_range=False)


            memory_model, encoder, decoder = build_model(sess, keys,
                                                         values,
                                                         FLAGS.num_copies,
                                                         FLAGS.minibatch_size)
            sess.run(tf.initialize_all_variables())

            # do a little validation on the keys
            # if FLAGS.complex_normalize and FLAGS.keytype != 'onehot':
            #     memory_model.verify_key_mod(keys)

            memories_host = encode(sess, memory_model, encoder, values, keys, minibatch, keys_host, FLAGS.minibatch_size)
            print 'memories recovered = ', memories_host.shape
            recovered_host = decode(sess, memory_model, decoder, values, keys, memories_host, keys_host, FLAGS.minibatch_size)

            for val, j in zip(recovered_host, range(len(recovered_host))):
                print 'vs = ', val.shape
                val = normalize(val)
                #np.savetxt("recovered_%d.csv" % j, val, delimiter=",")
                save_fig(val, "imgs/recovered_%d.png"  % j)
                print 'recovered value shape = ', val.shape
                #print 'recovered value [%s] = %s\n' % (val.shape, val)

if __name__ == "__main__":
    # Create our image directories
    if not os.path.exists('imgs'):
        os.makedirs('imgs')

    # Execute main loop
    main()
