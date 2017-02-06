import math
import tensorflow as tf
import numpy as np
from utils import tf_mean_std_normalize
from tensorflow.python.framework.tensor_shape import TensorShape

class HolographicMemory:
    def __init__(self, sess, input_size, num_models, seed=None,
                 complex_normalize=True, l2_normalize=False, use_fft_method=True):
        self.sess = sess
        self.input_size = input_size
        self.num_models = num_models
        self.complex_normalize = complex_normalize
        self.l2_normalize = l2_normalize
        self.conv_func = HolographicMemory.fft_circ_conv1d if use_fft_method \
                         else HolographicMemory.circ_conv1d

        # Perm dimensions are: num_models * [num_features x num_features]
        # Variables are used to store the results of the random values
        # as they need to be the same during recovery
        # self.perms = tf.pack([tf.Variable(self.create_permutation_matrix(input_size, seed+i if seed else None),
        #                                   trainable=False, name="perm_%d" % i)
        #                       for i in range(num_models)])
        self.perms = [self.create_permutation_matrix(input_size, seed+i if seed else None)
                      for i in range(num_models)]

        # Gather ND method
        # np.random.seed(seed if seed else None)
        # self.perms = [np.random.permutation(input_size) for _ in range(num_models)]
        # print 'perms = ', len(self.perms)

        # Random_Shuffle method
        # np.random.seed(seed if seed else None)
        # self.perms = [np.random.randint(9999999) for _ in range(num_models)]
        # print 'perms = ', len(self.perms)


    @staticmethod
    def _get_batch_perms(batch_size, perms):
        num_models = len(perms)
        input_size = perms[0].shape[0]
        perms_expanded = np.array([np.tile(p, batch_size) for p in perms]).flatten()
        print 'perms_expanded = ', perms_expanded, '| len = ', len(perms_expanded)
        x_inds = np.array(([[i]*input_size for i in range(batch_size)]*num_models)).flatten()
        return [[x, y] for x,y in zip(x_inds, perms_expanded)]

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
        return tf.complex_abs(xcplx)

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
        input_size = keys.get_shape().as_list()[1]
        assert input_size % 2 == 0, "input_size [%d] not divisible by 2" % input_size
        keys_mag = tf.maximum(tf.sqrt(tf.square(keys[:, 0:input_size/2])
                                      + tf.square(keys[:, input_size/2:])),
                              1.0)
        return keys / tf.concat(1, [keys_mag, keys_mag])

    @staticmethod
    def conj_real_by_complex(keys):
        reversed = tf.reverse(keys, [False, True])
        return tf.concat(1, [tf.expand_dims(keys[:, 0], 1),
                             reversed[:, 0:-1]])

    '''
    Accepts the already permuted keys and the data and encodes them

    keys: [num_models, num_features]
    X:    [batch_size, num_features]

    Returns: [num_models, num_features]
    '''
    @staticmethod
    def circ_conv1d(X, keys, batch_size, num_copies, num_keys=None, conj=False):
        if conj:
            keys = HolographicMemory.conj_real_by_complex(keys)

        # Get our original shapes
        xshp = X.get_shape().as_list()
        xshp[0] = batch_size if xshp[0] is None else xshp[0]
        kshp = keys.get_shape().as_list()
        kshp[0] = num_keys if num_keys is not None else kshp[0]
        kshp[1] = xshp[1] if kshp[1] is None else kshp[1]
        print 'X : ', xshp, ' | keys : ', kshp, ' | batch_size = ', batch_size

        # Concatenate X & keys
        num_dupes = kshp[0] / batch_size
        print 'num dupes = ', num_dupes
        xconcat = tf.tile(X, [num_dupes, 1]) \
                  if num_dupes > 1 else X
        xspshp = xconcat.get_shape().as_list()
        xspshp[0] = num_keys if xspshp[0] is None else xspshp[0]
        print 'xconcatinated : ', xspshp

        # The following computes all of the values individually, i.e
        # [P0k0 * x0, P0k1 * x1 + ...]
        # Input  : [batch, in_width, in_channels]
        # Filter : [filter_width, in_channels, out_channels]
        # Result : [batch, out_width, out_channels]
        conv = [tf.expand_dims(tf.squeeze(tf.nn.conv1d(tf.reshape(xconcat[i], [1, xspshp[1], 1]),
                                                       tf.reshape(keys[i], [kshp[1], 1, 1]),
                                                       stride=1,
                                                       padding='SAME')), 0) for i in range(kshp[0])]
        conv = tf.concat(0, conv)
        print 'conv = ', conv.get_shape().as_list()

        # We now aggregate them as follows:
        # c0 = P0k0 * x0 + P0k1 * x1 + ... P0k_batch * x_batch
        # and do that for all the c's and store separately
        #batch_size = xshp[0]
        batch_iter = min(batch_size, xshp[0]) if xshp[0] is not None else batch_size
        conv_concat = [tf.expand_dims(tf.reduce_mean(conv[begin:end], 0), 0)
                       for begin, end in zip(range(0, kshp[0], batch_iter),
                                             range(batch_iter, kshp[0]+1, batch_iter))]
        print 'conv concat = ', len(conv_concat), ' x ', conv_concat[0].get_shape().as_list()

        # return a single concatenated  tensor:
        # C = [c0; c1; ...]
        # C = tf.concat(0, conv_concat)
        # return (1.0 / C.get_shape().as_list()[0]) * C
        return tf.concat(0, conv_concat)

    @staticmethod
    def bound(x):
        bound = tf.maximum(tf.sqrt(tf.mul(tf.real(x), tf.real(x)) \
                                   + tf.mul(tf.imag(x), tf.imag(x))),
                           1.0)
        return tf.complex(tf.real(x) / bound, tf.imag(x) / bound)
    '''
    Does the entire operation within the frequency domain using
    ffts and element-wise matrix multiplies followed by reductions
    '''
    @staticmethod
    def fft_circ_conv1d(X, keys, batch_size, num_copies, num_keys=None, conj=False):
        if conj:
            keys = HolographicMemory.conj_real_by_complex(keys)

        # Get our original shapes
        xshp = X.get_shape().as_list()
        kshp = keys.get_shape().as_list()
        kshp[0] = num_keys if num_keys is not None else kshp[0]
        kshp[1] = xshp[1] if kshp[1] is None else kshp[1]
        print 'X : ', xshp, ' | keys : ', kshp, ' | batch_size = ', batch_size

        # duplicate out input data by the ratio: number_keys / batch_size
        # eg: |input| = [2, 784] ; |keys| = 3*[2, 784] ; (3 is the num_copies)
        #     |new_input| = 6/2 |input| = [input; input; input]
        #
        # At test: |memories| = [3, 784] ; |keys| = 3*[n, 784] ;
        #          |new_input| = 3n / 3 = n   [where n is the number of desired parallel retrievals]
        num_dupes = kshp[0] / batch_size
        print 'num dupes = ', num_dupes
        xcplx = HolographicMemory.split_to_complex(tf.tile(X, [num_dupes, 1]) \
                                                   if num_dupes > 1 else X)
        xshp = xcplx.get_shape().as_list()
        kcplx = HolographicMemory.split_to_complex(keys, kshp)

        # Convolve & re-cast to a real valued function
        unsplit_func = HolographicMemory.unsplit_from_complex_ri if not conj \
                       else HolographicMemory.unsplit_from_complex_ir
        #fft_mul = HolographicMemory.bound(tf.mul(tf.fft(xcplx), tf.fft(kcplx)))
        fft_mul = tf.mul(tf.fft(xcplx), tf.fft(kcplx))
        conv = unsplit_func(tf.ifft(fft_mul))
        print 'full conv = ', conv.get_shape().as_list()


        batch_iter = min(batch_size, xshp[0]) if xshp[0] is not None else batch_size
        print 'batch = ', batch_size, ' | num_copies = ', num_copies, '| num_keys = ', num_keys, \
            '| xshp[0] = ', xshp[0], ' | len(keys) = ', kshp[0], ' | batch iter = ', batch_iter
        conv_concat = [tf.expand_dims(tf.reduce_mean(conv[begin:end], 0), 0)
                       for begin, end in zip(range(0, kshp[0], batch_iter),
                                             range(batch_iter, kshp[0]+1, batch_iter))]
        print 'conv concat = ', len(conv_concat), ' x ', conv_concat[0].get_shape().as_list()

        # return a single concatenated  tensor:
        # C = [c0; c1; ...]
        C = tf.concat(0, conv_concat)

        return C
        #C = tf_mean_std_normalize(C)
        #return C / tf.maximum(tf.reduce_max(C), 1e-20)
        #return tf.nn.sigmoid(C)
        #return tf_mean_std_normalize(C)

    '''
    Helper to return the product of the permutation matrices and the keys

    K: [num_models, num_features]
    P: [num_models, feature_size, feature_size]
    '''
    @staticmethod
    # def perm_keys(K, P):
    #     # utilizes the random_shuffle method
    #     return tf.concat(0, [tf.transpose(tf.random_shuffle(tf.transpose(K), seed=s)) for s in P])


    def perm_keys(K, P, num_keys=None):
        # utilizes the sparse matmul method
        return tf.concat(0, [tf.transpose(tf.sparse_tensor_dense_matmul(P_i, K, adjoint_b=True))
                             for P_i in P])

    # def perm_keys(K, P):
    #     # utilizes the gather_nd method to permute
    #     kshp = K.get_shape().as_list()[1]
    #     print 'gathered = ', tf.gather_nd(K, P).get_shape().as_list()
    #     return tf.reshape(tf.gather_nd(K, P), [-1, kshp]) #tf.concat(0, [tf.reshape(tf.gather(K, p), kshp) for p in P])


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

    def _normalize(self, keys):
        # Normalize our keys to mod 1 if specified
        if self.complex_normalize:
            print 'normalizing via complex abs..'
            keys = HolographicMemory.normalize_real_by_complex_abs(keys)

        # Normalize our keys using the l2 norm
        if self.l2_normalize:
            print 'normalizing via l2..'
            keys = tf.nn.l2_normalize(keys, 1)

        return keys


    '''
    Encoders some keys and values together

    values: [batch_size, feature_size]
    keys:   [num_models, feature_size]
    perms:  [num_models, feature_size, feature_size]

    returns: [num_models, features]
    '''
    def encode(self, v, keys, batch_size=None):
        keys = self._normalize(keys)
        batch_size = v.get_shape().as_list()[0] if batch_size is None else batch_size
        permed_keys = self.perm_keys(keys, self.perms, num_keys=batch_size)
        print 'enc_perms = ', permed_keys.get_shape().as_list(), ' | batch_size = ', batch_size
        return self.conv_func(v, permed_keys,
                              batch_size,
                              self.num_models,
                              num_keys=batch_size*self.num_models)

    '''
    Decoders values out of memories

    memories: [num_models, feature_size]
    keys:     [num_models, feature_size]
    perms:    [num_models, feature_size, feature_size]

    returns: [num_models, features]
    '''
    def decode(self, memories, keys, num_keys=None):
        keys = self._normalize(keys)
        num_memories = memories.get_shape().as_list()
        num_memories[0] = self.num_models if num_memories[0] is None else num_memories[0]
        num_keys = keys.get_shape().as_list()[0] if num_keys is None else num_keys
        print 'decode: numkeys = ', num_keys, ' | num_memories = ', num_memories

        # re-gather keys to avoid mixing between different keys.
        perms = self.perm_keys(keys, self.perms, num_keys=num_keys)
        pshp = perms.get_shape().as_list()
        pshp[0] = num_keys*self.num_models if pshp[0] is None else pshp[0]
        pshp[1] = num_memories[1] if pshp[1] is None else pshp[1]
        permed_keys = tf.concat(0, [tf.strided_slice(perms, [i, 0], pshp, [num_keys, 1])
                                    for i in range(num_keys)])
        print 'memories = ', num_memories, \
            '| dec_perms =', permed_keys.get_shape().as_list()
        return self.conv_func(memories, permed_keys,
                              num_memories[0],
                              self.num_models,
                              num_keys=num_keys*self.num_models,
                              conj=True)

    '''
    Helper to create an [input_size, input_size] random permutation matrix
    '''
    @staticmethod
    def create_permutation_matrix(input_size, seed=None):
        #return tf.random_shuffle(tf.eye(input_size), seed=seed)
        ind = np.arange(0, input_size)
        ind_shuffled = np.copy(ind)
        np.random.seed(seed)
        np.random.shuffle(ind)
        indices = np.asarray([[x,y] for x,y in zip(ind, ind_shuffled)], dtype=np.int32)
        values = np.ones([len(indices)], dtype=np.float32)
        indices = indices[indices[:, 0].argsort()]
        return tf.SparseTensor(indices, values, shape=[input_size, input_size])

    '''
    Simple takes x and splits it in half --> Re{x[0:mid]} + Im{x[mid:end]}
    Works for batches in addition to single vectors
    '''
    @staticmethod
    def split_to_complex(x, xshp=None):
        xshp = x.get_shape().as_list() if xshp is None else xshp
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
        #return tf.concat(1, [tf.imag(x), tf.abs(tf.real(x))])
        return tf.abs(tf.concat(1, [tf.imag(x), tf.real(x)]))

        #mag = tf.maximum(1.0, tf.complex_abs(x))
        #x = tf.complex(tf.real(x) / (mag + 1e-10), tf.imag(x) / (mag + 1e-10))

        # real = tf.concat(1, [tf.imag(x), tf.real(x)])
        # return tf.abs(HolographicMemory.normalize_real_by_complex_abs([real])[0])
