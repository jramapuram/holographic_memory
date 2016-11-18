import math
import tensorflow as tf
import numpy as np

class HolographicMemory:
    def __init__(self, sess, input_size, num_models, seed=None):
        self.sess = sess
        self.input_size = input_size
        self.num_models = num_models
        if seed is None:
            seed = np.random.randint(0, 9999)

        # Perm dimensions are: num_models * [num_features x num_features]
        # Variables are used to store the results of the random values
        # as they need to be the same during recovery
        self.perms = [tf.Variable(self.create_permutation_matrix(input_size, seed+i), trainable=False)
                      for i in range(num_models)]

    '''
    Helper to decay the memories
    '''
    def update_hebb_weights(self, A, x, gamma=0.9):
        return gamma*A + tf.matmul(tf.transpose(x), x)

    '''
    Helper to do sqrt(sum(re(x_i)^2 + imag(x_i)^2))
    '''
    @staticmethod
    def complex_mod(x):
        return tf.abs(tf.sqrt(tf.reduce_sum(tf.square(tf.real(x)) + tf.square(tf.imag(x)))))

    '''
    Accepts the already permuted keys and the data and encodes them

    keys: [num_models, num_features]
    X:    [batch_size, num_features]

    Returns: [num_models, num_features]
    '''
    @staticmethod
    def circ_conv1d(X, keys, conj=False):
        fftx = tf.fft(HolographicMemory.split_to_complex(X))
        fftk = [tf.fft(HolographicMemory.split_to_complex(k)) for k in keys]
        if conj:
            fftk = [tf.conj(k) for k in fftk]

        fftmul = tf.concat(0, [HolographicMemory.unsplit_from_complex(tf.ifft(tf.mul(fk, fftx)))
                               for fk in fftk])
        return fftmul

    '''
    Helper to return the product of the permutation matrices and the keys

    K: [num_models, num_features]
    P: [num_models, feature_size, feature_size]
    '''
    @staticmethod
    def perm_keys(K, P):
        return [tf.matmul(k, p) for k,p in zip(K, P)]

    '''
    pads [batch, feature_size] --> [batch, feature_size + num_pad]
    '''
    @staticmethod
    def zero_pad(x, num_pad, index_to_pad=1):
        # Handle base case
        if num_pad == 0:
            return x

        xshp = x.get_shape().as_list()
        zeros = tf.zeros([xshp[0], num_pad]) if len(xshp) == 2 \
                else tf.zeros([num_pad])
        return tf.concat(index_to_pad, [x, zeros])

    '''
    returns [batch, in_width, in_channels]
    '''
    @staticmethod
    def _reshape_input(x):
        x_shp = x.get_shape().as_list()
        if len(x_shp) == 2:
            return tf.reshape(x, [x_shp[0], x_shp[1], 1])
        elif len(x_shp) == 1:
            return x
        else:
            raise Exception("unepexted number of dimensions in input x: %d" % len(x_shp))

    '''
    returns [filter_width, in_channels, out_channels]
    '''
    @staticmethod
    def _reshape_filter(v):
        v_shp = v.get_shape().as_list()
        if len(v_shp) == 2:
            # #return tf.reshape(v, [v_shp[0], v_shp[1], out_channels])
            # filters = tf.split(0, v_shp[0], v)
            # return [tf.reshape(f, [v_shp[1], 1, 1]) for f in filters]
            return tf.expand_dims(v, 2)
        else:
            return tf.reshape(v, [v_shp[0], 1, out_channels])

    '''
    Encoders some keys and values together

    values: [batch_size, feature_size]
    keys:   [num_models, feature_size]
    perms:  [num_models, feature_size, feature_size]

    returns: [num_models, features]
    '''
    def encode(self, v, keys):
        permed_keys = self.perm_keys(keys, self.perms)
        return self.circ_conv1d(v, permed_keys)

    '''
    Decoders values out of memories

    memories: [num_models, feature_size]
    keys:     [num_models, feature_size]
    perms:    [num_models, feature_size, feature_size]

    returns: [num_models, features]
    '''
    def decode(self, memories, keys):
        permed_keys = self.perm_keys(keys, self.perms)
        return self.circ_conv1d(memories, permed_keys, conj=True)

    '''
    Helper to create an [input_size, input_size] random permutation matrix
    '''
    @staticmethod
    def create_permutation_matrix(input_size, seed=None):
        np.random.seed(seed)
        ind = np.arange(0, input_size)
        ind_shuffled = np.copy(ind)
        np.random.shuffle(ind)
        retval = np.zeros([input_size, input_size])
        for x,y in zip(ind, ind_shuffled):
            retval[x, y] = 1

        return tf.constant(retval, dtype=tf.float32)

    '''
    Simple takes x and splits it in half --> Re{x[0:mid]} + Im{x[mid:end]}
    Works for batches in addition to single vectors
    '''
    @staticmethod
    def split_to_complex(x):
        xshp = x.get_shape().as_list()
        print 'split to complex shp = ', xshp
        if len(xshp) == 2:
            assert xshp[1] % 2 == 0, \
                "Vector is not evenly divisible into complex: %d" % xshp[1]
            mid = xshp[1] / 2
            return tf.complex(x[:, 0:mid], x[:, mid:])
        else:
            assert xshp[0] % 2 == 0, \
                "Vector is not evenly divisible into complex: %d" % xshp[0]
            mid = xshp[0] / 2
            return tf.complex(x[0:mid], x[mid:])

    '''
    Helper to un-concat (real, imag) --> single vector
    '''
    @staticmethod
    def unsplit_from_complex(x):
        return tf.concat(1, [tf.real(x), tf.imag(x)])
