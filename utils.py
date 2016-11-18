import os
import re
import string
import math
import random
import hashlib
import tarfile
import tensorflow as tf
import numpy as np
from copy import deepcopy
from tensorflow.python.framework import ops

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class BatchBuffer(object):
    def __init__(self, func, batch_size, buffer_size=10):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.func = func
        self.buffer = func(batch_size*buffer_size)
        self.remaining = buffer_size

    def get(self):
        if self.remaining == 0:
            self.buffer = self.func(self.batch_size*self.buffer_size)
            self.remaining = self.buffer_size

        start = len(self.buffer[0]) - self.remaining * self.batch_size
        end = start + self.batch_size
        retval = self.buffer[0][start:end], self.buffer[1][start:end]
        self.remaining -= 1
        return retval

def random_str(length):
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(length))

'''
Given a path and a regex string return the latest filename
Eg: r'vae(\d)+' will return vae(latestnumber)_...
TODO: this can probably be optimized to sort and start from the back, but meh
'''
def find_latest_file(path, regex):
    files_in_dir = os.listdir(path)
    latest = (None, 0)
    for f in files_in_dir:
        r = re.match(regex, f)
        current_index = int(r.group(1)) if r else 0
        print current_index, latest[1], current_index > latest[1]
        latest = (f, current_index) if current_index > latest[1] else latest

    return latest

def copy_to_graph(org_instance, to_graph, copied_variables={}, namespace=""):
    """
    Makes a copy of the Operation/Tensor instance 'org_instance'
    for the graph 'to_graph', recursively. Therefore, all required
    structures linked to org_instance will be automatically copied.
    'copied_variables' should be a dict mapping pertinent copied variable
    names to the copied instances.

    The new instances are automatically inserted into the given 'namespace'.
    If namespace='', it is inserted into the graph's global namespace.
    However, to avoid naming conflicts, its better to provide a namespace.
    If the instance(s) happens to be a part of collection(s), they are
    are added to the appropriate collections in to_graph as well.
    For example, for collection 'C' which the instance happens to be a
    part of, given a namespace 'N', the new instance will be a part of
    'N/C' in to_graph.

    Returns the corresponding instance with respect to to_graph.

    TODO: Order of insertion into collections is not preserved
    """

    #The name of the new instance
    if namespace != '':
        new_name = namespace + '/' + org_instance.name
    else:
        new_name = org_instance.name

    print 'new name = ', new_name

    #If a variable by the new name already exists, return the
    #correspondng tensor that will act as an input
    if new_name in copied_variables:
        return to_graph.get_tensor_by_name(
            copied_variables[new_name].name)

    #If an instance of the same name exists, return appropriately
    try:
        already_present = to_graph.as_graph_element(new_name,
                                                    allow_tensor=True,
                                                    allow_operation=True)
        return already_present
    except:
        pass

    #Get the collections that the new instance needs to be added to.
    #The new collections will also be a part of the given namespace.
    collections = []
    for name, collection in org_instance.graph._collections.items():
        if org_instance in collection:
            if namespace == '':
                collections.append(name)
            else:
                collections.append(namespace + '/' + name)

    #Take action based on the class of the instance

    if isinstance(org_instance, tf.python.framework.ops.Tensor):

        #If its a Tensor, it is one of the outputs of the underlying
        #op. Therefore, copy the op itself and return the appropriate
        #output.
        op = org_instance.op
        new_op = copy_to_graph(op, to_graph, copied_variables, namespace)
        output_index = op.outputs.index(org_instance)
        new_tensor = new_op.outputs[output_index]
        #Add to collections if any
        for collection in collections:
            to_graph.add_to_collection(collection, new_tensor)

        return new_tensor

    elif isinstance(org_instance, tf.python.framework.ops.Operation):

        op = org_instance

        #If it has an original_op parameter, copy it
        if op._original_op is not None:
            new_original_op = copy_to_graph(op._original_op, to_graph,
                                            copied_variables, namespace)
        else:
            new_original_op = None

        #If it has control inputs, call this function recursively on each.
        new_control_inputs = [copy_to_graph(x, to_graph, copied_variables,
                                            namespace)
                              for x in op.control_inputs]

        #If it has inputs, call this function recursively on each.
        new_inputs = [copy_to_graph(x, to_graph, copied_variables,
                                    namespace)
                      for x in op.inputs]

        #Make a new node_def based on that of the original.
        #An instance of tensorflow.core.framework.graph_pb2.NodeDef, it
        #stores String-based info such as name, device and type of the op.
        #Unique to every Operation instance.
        new_node_def = deepcopy(op._node_def)
        #Change the name
        new_node_def.name = new_name

        #Copy the other inputs needed for initialization
        output_types = op._output_types[:]
        input_types = op._input_types[:]

        #Make a copy of the op_def too.
        #Its unique to every _type_ of Operation.
        op_def = deepcopy(op._op_def)

        #Initialize a new Operation instance
        new_op = tf.python.framework.ops.Operation(new_node_def,
                                                   to_graph,
                                                   new_inputs,
                                                   output_types,
                                                   new_control_inputs,
                                                   input_types,
                                                   new_original_op,
                                                   op_def)
        #Use Graph's hidden methods to add the op
        to_graph._add_op(new_op)
        to_graph._record_op_seen_by_control_dependencies(new_op)
        for device_function in reversed(to_graph._device_function_stack):
            new_op._set_device(device_function(new_op))

        return new_op

    else:
        raise TypeError("Could not copy instance: " + str(org_instance))

def copy_variable_to_graph(org_instance, to_graph, namespace,
                           copied_variables={}):
    """
    Copies the Variable instance 'org_instance' into the graph
    'to_graph', under the given namespace.
    The dict 'copied_variables', if provided, will be updated with
    mapping the new variable's name to the instance.
    """

    if not isinstance(org_instance, tf.Variable):
        raise TypeError(str(org_instance) + " is not a Variable")

    #The name of the new variable
    if namespace != '':
        new_name = (namespace + '/' +
                    org_instance.name[:org_instance.name.index(':')])
    else:
        new_name = org_instance.name[:org_instance.name.index(':')]

    #Get the collections that the new instance needs to be added to.
    #The new collections will also be a part of the given namespace,
    #except the special ones required for variable initialization and
    #training.
    collections = []
    for name, collection in org_instance.graph._collections.items():
        if org_instance in collection:
            if (name == ops.GraphKeys.VARIABLES or
                name == ops.GraphKeys.TRAINABLE_VARIABLES or
                namespace == ''):
                collections.append(name)
            else:
                collections.append(namespace + '/' + name)

    #See if its trainable.
    trainable = (org_instance in org_instance.graph.get_collection(
        ops.GraphKeys.TRAINABLE_VARIABLES))
    #Get the initial value
    with org_instance.graph.as_default():
        temp_session = tf.Session()
        init_value = temp_session.run(org_instance.initialized_value())

    #Initialize the new variable
    with to_graph.as_default():
        new_var = tf.Variable(init_value,
                              trainable,
                              name=new_name,
                              collections=collections,
                              validate_shape=False)

    #Add to the copied_variables dict
    copied_variables[new_var.name] = new_var

    return new_var

def get_copied(original, graph, copied_variables={}, namespace=""):
    """
    Get a copy of the instance 'original', present in 'graph', under
    the given 'namespace'.
    'copied_variables' is a dict mapping pertinent variable names to the
    copy instances.
    """

    #The name of the copied instance
    if namespace != '':
        new_name = namespace + '/' + original.name
    else:
        new_name = original.name

    #If a variable by the name already exists, return it
    if new_name in copied_variables:
        return copied_variables[new_name]

    return graph.as_graph_element(new_name, allow_tensor=True,
                                  allow_operation=True)

def linear(input_, output_size, activation=None, scope=None,
           bias_init=0.0, with_params=False, reuse=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear", reuse=reuse):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_init))

        if activation is None:
            result = tf.matmul(input_, matrix) + bias
        else:
            result = activation(tf.matmul(input_, matrix) + bias)

        return [result, matrix, bias] if with_params else result

def hash_list(str_list):
    h = hashlib.new('ripemd160')
    for val in str_list:
        h.update(str(val))

    return h.hexdigest()

def unit_scale(t):
    m, v = tf.nn.moments(t, axes=[0])
    return  (t - m)/(v + 1e-9)

# returns a matrix with [num_rows, num_cols] where the indices value is 1
def one_hot(num_cols, indices):
    num_rows = len(indices)
    mat = np.zeros((num_rows, num_cols))
    mat[np.arange(num_rows), indices] = 1
    return mat

# 2x2 conv [downsample]
# F = [2, 2, 1, 32] w/ S=[1,2,2,1]
def conv_relu_2x2(x, W, b, filter_size, is_training, scope='bn'):
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME', use_cudnn_on_gpu=True) + b
    #bn = batch_norm(conv, filter_size, is_training, scope=scope)
    #return tf.nn.relu(bn)
    return tf.nn.relu(conv)

# 1x1 conv [shape preserving]
def conv_relu_1x1(x, W, b, filter_size, is_training, scope='bn'):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True) + b
    # bn = batch_norm(conv, filter_size, is_training, scope=scope)
    # return tf.nn.relu(bn)
    return tf.nn.relu(conv)

# 2d max pool [downsample]
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# 1d max pool [downsample]
def max_pool_1x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                          strides=[1, 1, 2, 1], padding='SAME')

# # 1d max pool [upsample]
# def max_pool_2x1(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],
#                           strides=[1, 1, 1, 1], padding='SAME')

# a debug helper to print out the tensor
def tensor_printer(tensor, name='tensor', summary_len=100):
    summary_str = name + ' = '
    return tf.Print(tensor, [tensor], summary_str, summarize=summary_len)

# filters labels by removing items that are in blacklist
# returns the tuple of images, labels that match filtration above
def zip_filter_unzip(images, labels, blacklist):
    return zip(*([im, lbl]
                 for im, lbl in zip(images, labels)
                 if lbl not in blacklist))

def xavier_init(n_inputs, n_outputs, uniform=True):
    """Set the parameter initialization using the method described.
    This method is designed to keep the scale of the gradients roughly the same
    in all layers.
    Xavier Glorot and Yoshua Bengio (2010):
        Understanding the difficulty of training deep feedforward neural
        networks. International conference on artificial intelligence and
        statistics.
    Args:
        n_inputs: The number of input nodes into each output.
        n_outputs: The number of output nodes for each input.
        uniform: If true use a uniform distribution, otherwise use a normal.
    Returns:
        An initializer.
  """
    if uniform:
        # 6 was used in the paper.
        init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
        initializer = tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
        initializer = tf.truncated_normal_initializer(stddev=stddev)

    return initializer([n_inputs, n_outputs], dtype=tf.float32)

def compress(output_file, sources):
    tar = tarfile.open(output_file, "w:gz")
    if type(sources) != list:
        sources = [sources]

    for name in sources:
        tar.add(name)

    tar.close()

def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y
