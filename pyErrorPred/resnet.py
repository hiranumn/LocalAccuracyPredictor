import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np

# Constructs a graph of resnet block
# Default input is channles last.
def resnet_block(_input,
                 isTraining,
                 channel=128,
                 require_bn=False, #Whether you need bn or not.
                 dilation_rate=(1,1),
                 data_format="channels_last"):
    
    if channel%2 != 0:
        print("Even number channels are required.")
        return -1
    down_channel = channel/2
    
    # BatchNorm if needed
    if require_bn: _input = tf.layers.batch_normalization(_input,
                                                          training=isTraining,
                                                          center=True,
                                                          scale=True,
                                                          axis=3)
    # Non linearity
    _input = tf.nn.elu(_input)
    # Projection down to 64 dims
    _input = tf.layers.conv2d(_input,
                              filters=down_channel,
                              kernel_size=(1,1),
                              data_format=data_format)
    
    # BatchNorm if needed
    if require_bn: _input = tf.layers.batch_normalization(_input,
                                                          training=isTraining,
                                                          center=True,
                                                          scale=True,
                                                          axis=3)
    # Non linearity
    _input = tf.nn.elu(_input)
    # 3 by 3 dialated convlolution with increasing dialation rate.
    _input = tf.layers.conv2d(_input,
                              filters=down_channel, 
                              kernel_size=(3,3), 
                              dilation_rate=dilation_rate,
                              data_format=data_format,
                              padding="same")
    
    # BatchNorm if needed
    if require_bn: _input = tf.layers.batch_normalization(_input,
                                                          training=isTraining,
                                                          center=True,
                                                          scale=True,
                                                          axis=3)
    # Non linearity
    _input = tf.nn.elu(_input)
    # Projection up to 128 dims.
    _input = tf.layers.conv2d(_input,
                              filters=channel, 
                              kernel_size=(1,1),
                              data_format=data_format)
    return _input

# Creates a resnet architecture.
def build_resnet(_input,
                 channel,
                 num_chunks,
                 isTraining,
                 require_bn=False, #Whether you need bn or not.
                 data_format="channels_last",
                 first_projection=True
                 ):
    
    # Projection of the very first input to 128 channels.
    if first_projection:
        _input = tf.layers.conv2d(_input,
                                  filters=channel,
                                  kernel_size=(1,1),
                                  dilation_rate=(1,1),
                                  data_format=data_format)
    
    # each chunk contatins 4 blocks with cycling dilation rates.
    for i in range(num_chunks):
        # dilation rates
        for dr in [1,2,4,8]:
            # save residual connection
            _residual = _input
            # pass through resnet block
            _conved = resnet_block(_input,
                                   isTraining,
                                   channel=channel,
                                   dilation_rate=(dr, dr),
                                   data_format=data_format)
            # genearte input to the next block
            _input = _residual+_conved
            
    return _input