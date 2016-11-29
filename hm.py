import math
import tensorflow as tf
import numpy as np
from tensorflow.python.framework.tensor_shape import TensorShape

class HolographicMemory:
    def __init__(self, sess, input_size, batch_size, num_models, seed=None, use_fft_method=False):
        self.sess = sess
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_models = num_models
        self.conv_func = HolographicMemory.fft_circ_conv1d if use_fft_method \
                         else HolographicMemory.circ_conv1d

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

        # Get our original shapes
        xshp = X.get_shape().as_list()
        kshp = keys[0].get_shape().as_list()
        print 'X : ', xshp, ' | keys : ', len(keys), ' x ', kshp

        # Concatenate X & keys
        xconcat = tf.concat(0, [X for _ in range(num_copies)]) \
                  if len(keys) > num_copies else X
        xspshp = xconcat.get_shape().as_list()
        kconcat = tf.concat(0, keys)
        print 'X_concats = ', xconcat.get_shape().as_list(), \
            'key_concats = ', kconcat.get_shape().as_list()

        # The following computes all of the values individually, i.e
        # [P0k0 * x0, P0k1 * x1 + ...]
        # Input  : [batch, in_width, in_channels]
        # Filter : [filter_width, in_channels, out_channels]
        # Result : [batch, out_width, out_channels]
        conv_0 = [tf.constant(0, dtype=tf.int32), tf.zeros([1, xshp[1]])]
        cond = lambda i, kx: tf.less(i, len(keys))
        _update = lambda i, kx_pair: \
                  [i+1, tf.concat(0, [kx_pair, tf.expand_dims(tf.squeeze(tf.nn.conv1d(tf.reshape(xconcat[i], [1, xshp[1], 1 if len(xshp) == 2 else xshp[0]]),
                                                                                      tf.reshape(kconcat[i], [kshp[1], kshp[0], 1]),
                                                                                      stride=1,
                                                                                      padding='SAME')), 0)])]
        _, conv = tf.while_loop(cond, _update, conv_0,
                                shape_invariants=[conv_0[0].get_shape(),
                                                  TensorShape([None, xshp[1]])],
                                parallel_iterations=len(keys))
        conv = conv[1:] # The 0th element is zeros(1, xshp[1])

        # We now aggregate them as follows:
        # c0 = P0k0 * x0 + P0k1 * x1 + ... P0k_batch * x_batch
        # and do that for all the c's and store separately
        #batch_size = xshp[0]
        batch_iter = min(batch_size, xspshp[0])
        conv_concat = [tf.expand_dims(tf.reduce_sum(conv[begin:end], 0), 0)
                       for begin, end in zip(range(0, len(keys), batch_iter),
                                             range(batch_iter, len(keys)+1, batch_iter))]
        print 'conv concat = ', len(conv_concat), ' x ', conv_concat[0].get_shape().as_list()

        # return a single concatenated  tensor:
        # C = [c0; c1; ...]
        return tf.concat(0, conv_concat)

    '''
    Does the entire operation within the frequency domain using
    ffts and element-wise matrix multiplies followed by reductions
    '''
    @staticmethod
    def fft_circ_conv1d(X, keys, batch_size, num_copies, conj=False):
        assert len(keys) > 0
        if conj:
            keys = HolographicMemory.conj_real_by_complex(keys)

        X = HolographicMemory.split_to_complex(X)
        xshp = X.get_shape().as_list()

        # setup keys
        keys_concat = [tf.concat(0, keys[begin:end], name='_'.join([k.name.replace(":0", "") for k in keys[begin:end]]))
                       for begin, end in zip(range(0, len(keys), min(batch_size, len(keys))),
                                             range(min(batch_size, len(keys)), len(keys)+1, min(batch_size, len(keys))))]
        # xmid = xspshp[1] / 2
        # kmid = kbshp[1] / 2
        # cplxx = tf.fft(tf.complex(xsplits[:, 0:xmid], xsplits[:, xmid:]))
        # cplxk = tf.fft(tf.complex(key_block[:, 0:kmid], key_block[:, kmid:]))
        # print 'x = ', cplxx.get_shape().as_list(), ' | k = ', cplxk.get_shape().as_list()
        # fft = tf.ifft(tf.mul(cplxk, cplxx))
        # print 'fft = ', fft.get_shape().as_list()
        # rec = tf.concat(1, [tf.real(fft), tf.imag(fft)])
        # print 'rec = ', rec.get_shape().as_list()
        # conv = tf.reshape(tf.concat(1, [tf.real(fft), tf.imag(fft)]), [-1, xshp[1]])

        #keys_concat = tf.concat(0, keys)
        #print 'keys0 : ', keys_concat.get_shape().as_list()
        keys_concat = [HolographicMemory.split_to_complex(k) for k in keys_concat]
        kcshp = keys_concat[0].get_shape().as_list()
        print 'x : ', xshp, ' | keys : ', len(keys_concat), 'x', kcshp

        test = [tf.ifft(tf.mul(tf.fft(k), tf.fft(X)))
                # for begin, end in zip(range(0, kcshp[0], min(batch_size, len(keys))),
                #                       range(min(batch_size, len(keys)), kcshp[0]+1, min(batch_size, len(keys))))]
                for k in keys_concat]
        print 'test = ', len(test), 'x', test[0].get_shape().as_list()

        conv = [tf.expand_dims(tf.reduce_sum(tf.ifft(tf.mul(tf.fft(k), tf.fft(X))), 0), 0)
                for k in keys_concat]
                # for begin, end in zip(range(0, kcshp[0], min(batch_size, len(keys))),
                #                       range(min(batch_size, len(keys)), kcshp[0]+1, min(batch_size, len(keys))))]
        cshp = conv[0].get_shape().as_list()
        print 'conv = ', len(conv), 'x', cshp

        if conj:
            result = [HolographicMemory.unsplit_from_complex_ir(c) for c in conv]
        else:
            result = [HolographicMemory.unsplit_from_complex_ri(c) for c in conv]
        # result = [tf.expand_dims(tf.reduce_sum(r[begin:end], 0), 0)
        #                          for begin, end in zip(range(0, len(conv), min(batch_size, len(conv))),
        #                                                range(min(batch_size, len(conv)), len(conv)+1, min(batch_size, len(conv))))]
        print 'result = ', len(result), 'x', result[0].get_shape().as_list()
        return tf.concat(0, result)


    '''
    Helper to return the product of the permutation matrices and the keys

    K: [num_models, num_features]
    P: [num_models, feature_size, feature_size]
    '''
    @staticmethod
    def perm_keys(K, P):
        return [tf.matmul(k, p, name="%s_%s" % (p.name.replace(":0", "")
                                                , k.name.replace(":0", "")))
                for p in P for k in K]

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
       return self.conv_func(v, permed_keys, self.batch_size, self.num_models)

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
        return self.conv_func(memories, permed_keys,
                              memories.get_shape().as_list()[0],
                              self.num_models, conj=True)

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
    def unsplit_from_complex_ri(x):
        return tf.concat(1, [tf.real(x), tf.imag(x)])

    '''
    Helper to un-concat (imag, real) --> single vector
    '''
    @staticmethod
    def unsplit_from_complex_ir(x):
        return tf.concat(1, [tf.imag(x), tf.real(x)])
