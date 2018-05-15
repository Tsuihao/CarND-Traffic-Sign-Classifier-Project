import tensorflow as tf
from tensorflow.contrib.layers import flatten

#TODO: 
# 1. Add namespace for tensorboard visualization

def cnn_arc_1(X, keep_prob=1.0):
    """
    6 layers of CNN structure, referenced from LetNet. However, based on the model technics, 
    * Reduce the patch/filter/kernel size
    * Add dropout in the FC layers
    * One conv more than LetNet   
    conv1:
    conv2:
    conv3:
    FC4
    FC5
    FC6
    """
    mu = 0
    sigma = 0.1
    
    # ========================= Conv_1 =================================
    # Input: 32x32x1  output: 30x30x10
    w_1 = tf.Variable(tf.truncated_normal((3,3,1,10), mean=mu, stddev=sigma))
    b_1 = tf.Variable(tf.zeros(10))
    conv_1 = tf.nn.conv2d(X, w_1, strides=[1,1,1,1], padding='VALID') + b_1
    conv_1 = tf.nn.relu(conv_1)
    # Input: 30x30x10 output: 15x15x10
    pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # ========================= Conv_2 =================================
    # Input: 15x15x10 output: 13*13*30
    w_2 = tf.Variable(tf.truncated_normal((3,3,10,30), mu, sigma))
    b_2 = tf.Variable(tf.zeros(30))
    conv_2 = tf.nn.conv2d(pool_1, w_2, strides=[1,1,1,1], padding='VALID') + b_2
    conv_2 = tf.nn.relu(conv_2)
    # Input: 13*13*30 output: 6x6x30
    pool_2 = tf.nn.max_pool(conv_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID') # TBD The pooling is bad
    
    
    # ========================= Conv_3 =================================
    # Input: 6x6x30 output: 4x4x60
    w_3 = tf.Variable(tf.truncated_normal((3,3,30,60), mu, sigma))
    b_3 = tf.Variable(tf.zeros(60))
    conv_3 = tf.nn.conv2d(pool_2, w_3, strides=[1,1,1,1], padding='VALID')
    conv_3 = tf.nn.relu(conv_3)
    # Input: 4x5460 output:2x2x60
    pool_3 = tf.nn.max_pool(conv_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID') #TBD
    
    # Input: 2x2x60 output: 240
    flatten_3 = flatten(pool_3)
    
    # ========================== FC4 =====================================
    # Input: 240, output:160
    w_4 = tf.Variable(tf.truncated_normal((240, 160), mu, sigma))
    b_4 = tf.Variable(tf.zeros(160))
    FC_4 = tf.add(tf.matmul(flatten_3, w_4), b_4)
    FC_4 = tf.nn.relu(FC_4)
    FC_4 = tf.nn.dropout(FC_4, keep_prob) # dropout
    
    # ========================== FC5 ======================================
    # Input: 160, output:84
    w_5 = tf.Variable(tf.truncated_normal((160, 84), mu, sigma))
    b_5 = tf.Variable(tf.zeros(84))
    FC_5 = tf.add(tf.matmul(FC_4, w_5), b_5)
    FC_5 = tf.nn.relu(FC_5)
    FC_5 = tf.nn.dropout(FC_5, keep_prob) # dropout
      
    # ========================== FC6 ======================================
    # Input: 84 output:43
    w_6 = tf.Variable(tf.truncated_normal((84,43), mu, sigma))
    b_6 = tf.Variable(tf.zeros(43))
    FC_6 = tf.add(tf.matmul(FC_5, w_6), b_6)
    logits = FC_6
    
    return logits


def cnn_arc_2(X, keep_prob=1.0):
    NotImplemented
    """
    Reference from the LetNet
    conv_1
    conv_2
    FC_3
    FC_4
    FC_5
    """
    
    
def cnn_arc_3(X, keep_prob=1.0):
    NotImplemented
    """
    Reference from the VGG16
    """
    