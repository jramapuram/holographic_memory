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
    Get complex mod of a real vector
    '''
    @staticmethod
    def complex_mod_of_real(x):
        xshp = x.get_shape().as_list()
        assert xshp[1] % 2 == 0
        xcplx = tf.complex(x[:, 0:xshp[1]/2], x[:, xshp[1]/2:])
        return tf.abs(xcplx)

    '''
    Helper to validate that all the keys have complex mod of 1.0
    '''
    def verify_key_mod(self, keys, print_l2=False):
        ops = []
        for k in [HolographicMemory.complex_mod_of_real(k) for k in keys]:
            ops.append(tf.nn.l2_loss(k - tf.ones_like(k)))

        keys_l2 = self.sess.run(ops)
        for l2, k in zip(keys_l2, keys):
            assert l2 < 1e-9, "key [%s] is not normalized, l2 = %f \n%s" \
                % (k, l2, str(self.sess.run(k)))
            if print_l2:
                print 'l2 = ', l2

        print '|keys| ~= 1.0: verified'


    '''
    Normalizes real valued keys to have complex abs of 1.0

    keys: f32/f64 list of keys of [1, input_size]

    Returns: list of [1, input_size] f32/f64
    '''
    @staticmethod
    def normalize_real_by_complex_abs(keys):
        assert len(keys) > 0
        input_size = keys[0].get_shape().as_list()[1]
        assert input_size % 2 == 0, "input_size [%d] not divisible by 2" % input_size
        keys_mag = [tf.sqrt(tf.square(k[:, 0:input_size/2])
                            + tf.square(k[:, input_size/2:])) for k in keys]
        keys_mag = [tf.concat(1, [km, km]) for km in keys_mag]
        return [k / (km + 1e-10) for k, km in zip(keys, keys_mag)]

    '''
    Accepts the already permuted keys and the data and encodes them

    keys: [num_models, num_features]
    X:    [batch_size, num_features]

    Returns: [num_models, num_features]
    '''
    @staticmethod
    def circ_conv1d(X, keys, conj=False):
        xnorm = [X] #HolographicMemory.normalize_real_by_complex_abs([X])
        fftx = tf.fft(HolographicMemory.split_to_complex(xnorm[0]))
        fftk = [tf.fft(HolographicMemory.split_to_complex(k)) for k in keys]
        if conj:
            fftk = [tf.conj(k) for k in fftk]

        print 'fftx : ', fftx.get_shape().as_list(), ' | fftk : ', len(fftk), \
            ' x ', fftk[0].get_shape().as_list()
        fftmul = tf.concat(0, [tf.expand_dims(tf.reduce_sum(HolographicMemory.unsplit_from_complex(tf.ifft(tf.mul(fk, fftx))), 0), 0)
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
        return tf.reduce_mean(self.circ_conv1d(memories, permed_keys, conj=True), 0)

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
