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
        self.perms = [tf.Variable(self.create_permutation_matrix(input_size, seed+i),
                                  trainable=False, name="perm_%d" % i)
                      for i in range(num_models)]
        # self.perms = [tf.Variable(np.identity(input_size), dtype=tf.float32,
        #                           trainable=False, name="perm_%d" % i)
        #               for i in range(num_models)]


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
    # @staticmethod
    # def circ_conv1d(X, keys, conj=False):
    #     xnorm = [X] #HolographicMemory.normalize_real_by_complex_abs([X])
    #     xshp = X.get_shape().as_list()
    #     fftx = tf.fft(HolographicMemory.split_to_complex(xnorm[0]))
    #     fftk = [tf.fft(HolographicMemory.split_to_complex(k)) for k in keys]
    #     if conj:
    #         fftk = [tf.conj(k) for k in fftk]

    #     print 'fftx : ', fftx.get_shape().as_list(), ' | fftk : ', len(fftk), \
    #         ' x ', fftk[0].get_shape().as_list()

    #     fftmul = tf.concat(0, [tf.expand_dims(tf.reduce_sum(HolographicMemory.unsplit_from_complex(tf.ifft(tf.mul(fk, fftx))), 0), 0)
    #                            for fk in fftk])
    #     C = [tf.expand_dims(tf.reduce_sum(fftmul[begin:end], 0), 0)
    #          for begin, end in zip(range(0, len(keys), xshp[0]),
    #                                range(xshp[0], len(keys)+1, xshp[0]))]
    #     print 'C : ', len(C), C[0].get_shape().as_list()
    #     return tf.concat(0, C)

    @staticmethod
    def circ_conv1d(X, keys, conj=False):
        assert len(keys) > 0
        if conj:
            keys = HolographicMemory.conj_real_by_complex(keys)

        xshp = X.get_shape().as_list()
        kshp = keys[0].get_shape().as_list()
        print 'x : ', xshp, ' | keys : ', len(keys), ' x ', kshp

        # Input  : [batch, in_width, in_channels]
        # Filter : [filter_width, in_channels, out_channels]
        # Result : [batch, out_width, out_channels]
        # conv = [tf.nn.conv1d(tf.reshape(X, xshp + [1]),
        #                      tf.reshape(k, [kshp[1], kshp[0], 1]),
        #                      stride=1,
        #                      padding='SAME') for k in keys]
        # # add up elements
        # conv_filtered = [tf.expand_dims(tf.reduce_sum(conv[begin:end], 0), 0)
        #                  for begin, end in zip(range(0, len(keys), xshp[0]),
        #                                        range(xshp[0], len(keys)+1, xshp[0]))]
        # print 'conv = ', len(conv_filtered), ' x ', conv_filtered[0].get_shape().as_list()

        # C = tf.squeeze(tf.concat(0, conv_filtered), axis=[2])
        # print 'C: ', C.get_shape().as_list()
        # return C

        #keys_concat = tf.concat(0, keys)

        # keys_concat = [tf.concat(0, keys[begin:end])
        #                for begin, end in zip(range(0, len(keys), xshp[0]),
        #                                      range(xshp[0], len(keys)+1, xshp[0]))]

        batch_size = xshp[0]
        keys_concat = [tf.concat(0, keys[begin:end])
                       for begin, end in zip(range(0, len(keys), batch_size),
                                             range(batch_size, len(keys)+1, batch_size))]

        kcshp = keys_concat[0].get_shape().as_list()
        print 'concat keys : ', len(keys_concat), ' x  ', kcshp

        # conv = [tf.reduce_sum(tf.nn.conv1d(tf.reshape(X, xshp + [1]),
        #                                    tf.reshape(k, [kcshp[1], 1, kcshp[0]]),
        #                                    stride=1,
        #                                    padding='SAME'), 0) for k in keys_concat]

        # xsplits = tf.split(0, xshp[0], X)
        # xspshp = xsplits[0].get_shape().as_list()
        # print 'split shape = ', len(xsplits), 'x', xspshp
        # conv = [tf.nn.conv1d(tf.reshape(x, xspshp + [1]),
        #                      tf.reshape(k, [kshp[1], kshp[0], 1]),
        #                      stride=1,
        #                      padding='SAME') for k,x in zip(keys, xsplits)]
        # print 'conv = ', len(conv), ' x ', conv[0].get_shape().as_list()
        # return tf.concat(0, tf.transpose(conv))

        # conv = [tf.transpose(tf.reduce_sum(tf.nn.conv1d(tf.reshape(X, xshp + [1]),
        #                                                 tf.reshape(k, [kshp[1], kshp[0], 1]),
        #                                                 stride=1,
        #                                                 padding='SAME'), 0)) for k in keys]

        # test = [tf.reduce_sum(tf.nn.conv2d(tf.reshape(X, [xshp[0], 1, xshp[1], 1]),
        #                                    tf.reshape(k, [kcshp[0], kcshp[1], 1, 1]),
        #                                    strides=[1, 1, 1, 1],
        #                                    padding='SAME'), 0) for k in keys_concat]
        # print 'test = ', len(test), 'x', test[0].get_shape().as_list()

        conv = [tf.expand_dims(tf.squeeze(tf.reduce_sum(tf.nn.conv2d(tf.reshape(X, [xshp[0], 1, xshp[1], 1]),
                                                                     tf.reshape(k, [kcshp[0], kcshp[1], 1, 1]),
                                                                     strides=[1,1,1,1],
                                                                     padding='SAME'), 0)), 0) for k in keys_concat]

        print 'conv = ', len(conv), ' x ', conv[0].get_shape().as_list()
        return tf.concat(0, conv)




    '''
    Helper to return the product of the permutation matrices and the keys

    K: [num_models, num_features]
    P: [num_models, feature_size, feature_size]
    '''
    @staticmethod
    def perm_keys(K, P, conj=False):
        #return [tf.matmul(k, p) for k,p in zip(K, P)]
        # if conj:
        #     # print 'permkeys P shp = ', len(P), 'x', P[0].get_shape().as_list()
        #     P = HolographicMemory.conj_real_by_complex2(P)
        #     #P = [tf.transpose(p) for p in P]

        print 'p0 = ', P[0].get_shape().as_list()
        print 'k0 = ', K[0].get_shape().as_list()
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
        # conv = self.circ_conv1d(v, keys)
        # #return tf.concat(0, self.perm_keys(conv, self.perms))
        # batch_size = v.get_shape().as_list()[0]
        # return self._concat_reduce(self.perm_keys(conv, self.perms), batch_size)

       permed_keys = self.perm_keys(keys, self.perms)
       print 'enc_perms =', len(permed_keys), 'x', permed_keys[0].get_shape().as_list()
       # batch_size = v.get_shape().as_list()[0]
       # return self._concat_reduce(self.circ_conv1d(v, permed_keys), batch_size)
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
       print 'dec_perms =', len(permed_keys), 'x', permed_keys[0].get_shape().as_list()
       # batch_size = memories.get_shape().as_list()[0]
       # return self._concat_reduce(self.circ_conv1d(memories, permed_keys, conj=True), batch_size)
       return self.circ_conv1d(memories, permed_keys, conj=True)

        # conv = self.circ_conv1d(memories, keys, conj=True)
        # #return tf.concat(0, self.perm_keys(conv, self.perms, conj=True))
        # batch_size = memories.get_shape().as_list()[0]
        # return self._concat_reduce(self.perm_keys(conv, self.perms, conj=True), batch_size)

    def _concat_reduce(self, memories, batch_size):
        c = tf.concat(0, [tf.expand_dims(tf.reduce_sum(tf.concat(0, memories[begin:end]), 0), 0)
                             for begin, end in zip(range(0, len(memories), batch_size),
                                                   range(batch_size, len(memories)+1, batch_size))])
        print 'C = ', c.get_shape().as_list()
        return c

    '''
    Helper to create an [input_size, input_size] random permutation matrix
    '''
    @staticmethod
    def create_permutation_matrix(input_size, seed=None):
        #np.random.seed(seed)
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
