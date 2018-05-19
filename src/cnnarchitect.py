import pickle
import os
import numpy as np
import time
from sklearn.utils import shuffle
import random
import tensorflow as tf
from tensorflow.contrib.layers import flatten


class BaseClassifyNet:
    '''
    Base class for inheritance
    '''
    # constructor
    def __init__(self):
        self._load_data()
        self._placeholders()
        self.sess = None
        self._loss()
        self._optimize()
        self.init_variables = tf.global_variables_initializer()

    # private
    def _placeholder(self):
        raise NotImplementedError

    def _network_architect(self):
        raise NotImplementedError

    def _loss(self):
        raise NotImplementedError

    def _optimize(self):
        raise NotImplementedError

    def _evaluate(self):
        raise NotImplementedError

    def _load_data(self):
        raise NotImplementedError

    def _start_sess(self):
        raise NotImplementedError

    # public
    def train(self):
        raise NotImplementedError

    def evaluate_model(self):
        raise NotImplementedError

    det predict(self):
        raise NotImplementedError


class TrafficSignClassifier(BaseClassifyNet):
    '''
    Traffic Sign Calssifier Net derived from BaseClassifyNet
    '''

    # Constructor
    # Instanitate with config
    def __init__(self, config):
        self.config = config
        self.first_tuning_save_path = os.path.join(self.config.model_path, 'first_tuning_model')
        self.fine_tuning_save_path = os.path.join(self.config.model_path, 'fine_tuning_model')
        BaseClassifyNet.__init__(self) # call base class constructor

    def _placeholders(self):
        self.features = tf.placeholder(tf.float32, [None]+self.img_dims) # flexible that can be channel 1 or 3
        self.labels = tf.placeholder(tf.float32, [None, self.n_class])
        self.keep_prob = tf.placeholder(tf.float32)

    def _start_sess(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.saver = tf.train.Saver()

    def _reset_sess(self):
        if self.sess is not None:
            self.sess_close()
        self.sess = None

    def _load_data(self, pickle_fine=None):
        if pickle_file is None:
            pickle_file = self.config.pickle_file
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
            self.train_features = pickle_data['train_dataset']
            self.train_labels = pickle_data['train_labels']
            self.valid_features = pickle_data['valid_dataset']
            self.valid_labels = pickle_data['valid_labels']
            self.test_features = pickle_data['test_dataset']
            self.test_labels = pickle_data['test_labels']
            del pickle_data # Free memory

        self.img_dims = list(self.train_features.shape[1:]) # the first index is the number
        self.n_class = len(np.unique(train_labels))
        print('Train, valid, and test data are loaded from {}'.format(pickle_file))

    def _network_architect(self):
        def _conv_layer(name, input, kernel_size, stride=[1,1,1,1], padding='SAME',
                        max_pool=True, dropout=False, initializer=None):

            if initializer is None:
                w_initializer = tf.truncated_normal_initializer(mean=0, stddev=0.1)

            with tf.variable_scope(name):
                # Why tf.get_variable https://stackoverflow.com/questions/37098546/difference-between-variable-and-get-variable-in-tensorflow?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
                weight = tf.get_variable('conv_w', kernel_size, initializer=initializer)
                bias = tf.get_variable('conv_b', [kernel_size[-1]], initializer = tf.constant_initializer(0.0))

            conv = tf.nn.con2d(input, weights, stride, padding)
            result = tf.nn.relu(conv + bias)

            if dropout:
                result = tf.nn.dropout(result, keep_prob=self.keep_prob)

            if max_pool:
                result = tf.nn.max_pool(result, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

            return result

        def _fc_layer(name, input, n_out, dropout=False, initializer=None):
            n_in = input.get_shape().as_list()[-1]  # input shape [None, cols]
            if initializer is None:
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)

            with tf.variable_scope(name):
                weight = tf.get_variable('fc_w', [n_in, n_out], initializer=initializer)
                bias = tf.get_variable('fc_b', [n_out], initializer = tf.constant_initializer(0.0))
                fc = tf.matmul(input, weight) + bias

            if dropout:
                fc = tf.nn.dropout(fc, keep_prob=self.keep_prob)

            return fc

        def arc_1():
            conv1 = self._conv_layer('conv1', self.features, [3, 3, img_dims[-1], 10])
            conv2 = self._conv_layer('conv2', conv1, [3,3,10,30])
            conv3 = self._conv_layer('conv3', conv2, [3,3,30,60])
            flatten3 = flatten(conv3)
            fc4 = self._fc_layer('fc4', flatten3, 160, dropout=True)
            fc5 = self._fc_layer('fc5', fc4, 84, dropout=True)
            fc6 = self._fc_layer('fc6', fc5, n_class, dropout=False)
            logits = fc6

            return logits

        def _loss(self):
            self.logits = self._network_architect().arc_1() # Here choose different architecture
            self.predictions = tf.nn.softmax(self.logits)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.labels)
            # Can use different loss function, here we use the mean of cross entropy
            loss = tf.reduce_mean(cross_entropy)
            self.loss = loss

        def _optimize(self):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            self.optimize = optimizer.minimize(self.loss)

        def _evalutate(self, X, y):
            correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            n_samples = len(X)
            batch_size = self.config.batch_size
            total_accuracy = 0
            for offset in range(0, n_samples, batch_size):
                end = offset + batch_size
                X_batch, y_batch = X[offset:end], y[offset:end]
                acc = seLf.sess.run(accuracy, feed_dict={self.feautures: X_batch,
                                                         self.lables: y_batch,
                                                         self.keep_prob: 1.0}) # no dropout when evaluate
                total_accuracy += acc*len(X_batch)
                total_accuracy/n_samples
            return total_accuracy

        def evalute_model(self, X_, y_, model_path=None):
            """
            User need to specify the X_valid/X_test and y_valid/y_test
            If not given the model_path, will load the first_tuning_model by default
            """
            if (X_ is None) or (y_ is None):
                X_ = self.test_features
                y_ = self.test_labels
            assert(X_.shape[0] == y_.shpae[0])
            self._start_sess()

            if model_path is None:
                model_path = self.first_tuning_save_path

            assert(tf.train.checkpoint_exist(model_path) is True, model_path+ 'does not exist!')
            self.saver.restore(self.sess, model_path)
            acc = self._evaluate(X_,y_)

            print('Overall accuracy is {".3f"}%'.format(acc*100))

            self.test_all_labels_acc = []

            for i in range(self.n_class):
                indices = np.where(np.argmax(y, 1) == i)[0]
                X_i = X_[indices] # extract the specific class
                y_i = y_[indices]
                acc_i = self._evlauate(X_i, y_i)
                self.test_all_labels_acc.append(acc_i)
                print('class {}, test_num {}, acc = {:.3f}%'.format(i, len(indices), acc_i*100))
            self._reset_sess()

            def train(self, fine_tuning=False, first_tuning_model_path=None):
                '''
                Offer the fine_tuning
                '''
                self._start_sess()
                if fine_tuing:
                    if first_tuning_model_path is None:
                        model_path = self.first_tuning_model_path
                    assert(tf.train.get_checkpoint_exist(model_path) is True, model_path+ 'doest not exist!')
                    self.saver.restore(sess, model_path)

                else:
                    self.sess.run(self.init_variables)
                    EPOCHS = self.config.epochs
                    batch_size = self.config.batch_size
                    n_examples = len(self.train_features)

                    best_epoch = -1 # Note the best model_path
                    self.loss_batch = []
                    self.train_acc_batch = []
                    self.valid_acc_batch = []


                    print('Traing start..')
                    for epoch in range(EPOCHS):
                        start_time = time.time()

                        # Very important schffle!
                        self.train_features, self.train_labels = shuffle(self.train_features, self.train_labels)
                        for offset in range(0, batch_size, n_examples):
                            end = offset + batch_size
                            X_batch, y_batch = self.train_features[offset:end], self.train_labels[offest:end]
                            _, _loss = self.sess.run([self.optimize, self.loss], feed_dict:{self.features: X_batch
                                                                                           self.labels: y_batch
                                                                                           self.keep_prob: 0.5})
                           end_time = time.time()
                           speed = int(n_examples // (end_time - start_time))
                           self.loss_batch.append(_loss)
                           train_acc = self._evaluate(self.train_features, self.train_labels)
                           self.train_acc_batch.append(train_acc)
                           valid_acc = self._evalutate(self.valid_features, self.valid_labels)
                           self.valid_acc_batch.append(valid_acc)








































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

    with tf.name_scope('arc_1'):
        # ========================= Conv_1 =================================
        # Input: 32x32x1  output: 30x30x10
        w_1 = tf.Variable(tf.truncated_normal((3,3,1,10), mean=mu, stddev=sigma), name='w_1')
        b_1 = tf.Variable(tf.zeros(10), name='b_1')
        conv_1 = tf.nn.conv2d(X, w_1, strides=[1,1,1,1], padding='VALID') + b_1
        conv_1 = tf.nn.relu(conv_1, name='conv_1')
        # Input: 30x30x10 output: 15x15x10
        pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # ========================= Conv_2 =================================
        # Input: 15x15x10 output: 13*13*30
        w_2 = tf.Variable(tf.truncated_normal((3,3,10,30), mu, sigma), name='w_2')
        b_2 = tf.Variable(tf.zeros(30), name='b_2')
        conv_2 = tf.nn.conv2d(pool_1, w_2, strides=[1,1,1,1], padding='VALID') + b_2
        conv_2 = tf.nn.relu(conv_2, name='conv_2')
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
