import math
import tensorflow as tf
import numpy as np

class HolographicMemory:
    def __init__(self, sess, input_size, batch_size, num_models, seed=None):
        self.sess = sess
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_models = num_models

        # Perm dimensions are: num_models * [num_features x num_features]
        # Variables are used to store the results of the random values
        # as they need to be the same during recovery
        self.perms = [tf.Variable(self.create_permutation_matrix(input_size, seed+i if seed else None),
                                  trainable=False, name="perm_%d" % i)
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

    @staticmethod
    def conj_real_by_complex(keys):
        assert len(keys) > 0
        input_size = keys[0].get_shape().as_list()[1]
        indexes = tf.concat(0, [tf.constant([0]), tf.range(input_size-1, 0, -1)])
        return [tf.expand_dims(tf.gather(tf.squeeze(k, axis=[0]), indexes), 0) for k in keys]

    '''
    Accepts the already permuted keys and the data and encodes them

    keys: [num_models, num_features]
    X:    [batch_size, num_features]

    Returns: [num_models, num_features]
    '''
    @staticmethod
    def circ_conv1d(X, keys, batch_size, num_copies, conj=False):
        assert len(keys) > 0
        if conj:
            keys = HolographicMemory.conj_real_by_complex(keys)

        xshp = X.get_shape().as_list()
        kshp = keys[0].get_shape().as_list()
        print 'x : ', xshp, ' | keys : ', len(keys), ' x ', kshp

        # Split X's batches out
        xsplits = tf.split(0, xshp[0] * num_copies, tf.concat(0, [X for _ in range(num_copies)]))
        xspshp = xsplits[0].get_shape().as_list()
        print 'xsplits : ', len(xsplits), 'x', xspshp

        # The following computes all of the values individually, i.e
        # [P0k0 * x0, P0k1 * x1 + ...]
        # Input  : [batch, in_width, in_channels]
        # Filter : [filter_width, in_channels, out_channels]
        # Result : [batch, out_width, out_channels]
        conv = [tf.expand_dims(tf.squeeze(tf.nn.conv1d(tf.reshape(x, [xspshp[0], xspshp[1], 1]),
                                                       tf.reshape(k, [kshp[1], kshp[0], 1]),
                                                       stride=1,
                                                       padding='SAME')), 0) for k,x in zip(keys, xsplits)]
        print 'full conv list = ', len(conv), ' x ', conv[0].get_shape().as_list()

        # We now aggregate them as follows:
        # c0 = P0k0 * x0 + P0k1 * x1 + ... P0k_batch * x_batch
        # and do that for all the c's and store separately
        #batch_size = xshp[0]
        conv_concat = [tf.expand_dims(tf.reduce_sum(tf.concat(0, conv[begin:end]), 0), 0)
                       for begin, end in zip(range(0, len(conv), min(batch_size, len(conv))),
                                             range(min(batch_size, len(conv)), len(conv)+1, min(batch_size, len(conv))))]


        print 'conv concat = ', len(conv_concat), ' x ', conv_concat[0].get_shape().as_list()

        # return a single concatenated  tensor:
        # C = [c0; c1; ...]
        return tf.concat(0, conv_concat)

    # def circ_conv1d(X, keys, batch_size, num_copies, conj=False):
    #     assert len(keys) > 0
    #     if conj:
    #         keys = HolographicMemory.conj_real_by_complex(keys)

    #     #X = tf.concat(0, [X for _ in range(num_copies)])
    #     xshp = X.get_shape().as_list()
    #     kshp = keys[0].get_shape().as_list()
    #     print 'x : ', xshp, ' | keys : ', len(keys), ' x ', kshp

    #     keys_concat = [tf.concat(0, keys[begin:end])
    #                    for begin, end in zip(range(0, len(keys), min(batch_size, len(keys))),
    #                                          range(min(batch_size, len(keys)), len(keys)+1, min(batch_size, len(keys))))]
    #     print 'kc = ', len(keys_concat), 'x', keys_concat[0].get_shape().as_list()

    #     # Input:  [batch, in_height, in_width, in_channels]
    #     # Filter: [filter_height, filter_width, in_channels, out_channels]
    #     conv = [tf.expand_dims(tf.reduce_sum(tf.squeeze(tf.nn.conv2d(tf.reshape(X, [1, xshp[0], xshp[1], 1]),
    #                                                                  tf.reshape(k, [batch_size, kshp[1], 1, 1]),
    #                                                                  strides=[1,1,1,1],
    #                                                                  padding='SAME'),
    #                                                     [0, 3]), 0), 0) for k in keys_concat]
    #     print 'full conv list = ', len(conv), ' x ', conv[0].get_shape().as_list()
    #     # cshp = conv.get_shape().as_list()
    #     # print 'full conv list = ', cshp

    #     # # We now aggregate them as follows:
    #     # # c0 = P0k0 * x0 + P0k1 * x1 + ... P0k_batch * x_batch
    #     # # and do that for all the c's and store separately
    #     # #batch_size = xshp[0]
    #     # conv_concat = [tf.expand_dims(tf.squeeze(tf.reduce_sum(conv[begin:end], 0)), 0)
    #     #                for begin, end in zip(range(0, cshp[0], batch_size),
    #     #                                      range(batch_size, cshp[0]+1, batch_size))]


    #     # print 'conv concat = ', len(conv_concat), ' x ', conv_concat[0].get_shape().as_list()

    #     # # return a single concatenated  tensor:
    #     # # C = [c0; c1; ...]
    #     return tf.concat(0, conv)




    '''
    Helper to return the product of the permutation matrices and the keys

    K: [num_models, num_features]
    P: [num_models, feature_size, feature_size]
    '''
    @staticmethod
    def perm_keys(K, P):
        return [tf.matmul(k, p) for p in P for k in K]

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
    Encoders some keys and values together

    values: [batch_size, feature_size]
    keys:   [num_models, feature_size]
    perms:  [num_models, feature_size, feature_size]

    returns: [num_models, features]
    '''
    def encode(self, v, keys):
       permed_keys = self.perm_keys(keys, self.perms)
       print 'enc_perms =', len(permed_keys), 'x', permed_keys[0].get_shape().as_list()
       #total_possible_keys = self.num_models *
       return self.circ_conv1d(v, permed_keys, self.batch_size, self.num_models)

    '''
    Decoders values out of memories

    memories: [num_models, feature_size]
    keys:     [num_models, feature_size]
    perms:    [num_models, feature_size, feature_size]

    returns: [num_models, features]
    '''
    def decode(self, memories, keys):
        # if len(keys) < self.batch_size:
        #     keys = keys + [tf.zeros([1, self.input_size]) for _ in range(self.batch_size - len(keys))]

        permed_keys = self.perm_keys(keys, self.perms)
        print 'dec_perms =', len(permed_keys), 'x', permed_keys[0].get_shape().as_list()
        return self.circ_conv1d(memories, permed_keys, self.batch_size, self.num_models, conj=True)

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
