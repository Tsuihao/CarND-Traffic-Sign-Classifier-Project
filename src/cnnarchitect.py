import pickle
import os
import numpy as np
import time
from sklearn.utils import shuffle
import random
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import matplotlib.pyplot as plt


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

    def predict(self):
        raise NotImplementedError


class TrafficSignClassifier(BaseClassifyNet):
    '''
    Traffic Sign Calssifier Net derived from BaseClassifyNet
    '''

    # Constructor
    # Instanitate with config
    def __init__(self, config):
        self.config = config
        self.best_model_save_path = os.path.join(self.config.model_path, 'model')
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
            self.sess.close()
        self.sess = None

    def _load_data(self, pickle_file=None):
        if pickle_file is None:
            pickle_file = self.config.pickle_file # Load pickle
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
        self.n_class = self.train_labels.shape[-1]
        print('Train, valid, and test data are loaded from {}'.format(pickle_file))

    def _network_architect(self, config_arc):
        def _conv_layer(name, input, kernel_size, stride=[1,1,1,1], padding='SAME',
                        max_pool=True, dropout=False, initializer=None):

            if initializer is None:
                w_initializer = tf.truncated_normal_initializer(mean=0, stddev=0.1)

            with tf.variable_scope(name):
                # Why tf.get_variable https://stackoverflow.com/questions/37098546/difference-between-variable-and-get-variable-in-tensorflow?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
                weights = tf.get_variable(name+'_w', kernel_size, initializer=w_initializer)
                bias = tf.get_variable(name+'_b', [kernel_size[-1]], initializer = tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(input, weights, stride, padding)
            result = tf.nn.relu(conv + bias)

            if dropout:
                result = tf.nn.dropout(result, keep_prob=self.keep_prob)

            if max_pool:
                result = tf.nn.max_pool(result, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

            return result

        def _fc_layer(name, input, n_out, dropout=False, initializer=None):
            n_in = input.get_shape().as_list()[-1]  # input shape [None, cols]
            if initializer is None:
                w_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1)

            with tf.variable_scope(name):
                weights = tf.get_variable(name+'_w', [n_in, n_out], initializer=w_initializer)
                bias = tf.get_variable(name+'_b', [n_out], initializer = tf.constant_initializer(0.0))
            fc = tf.matmul(input, weights) + bias
            fc = tf.nn.relu(fc)
            
            if dropout:
                fc = tf.nn.dropout(fc, keep_prob=self.keep_prob)

            return fc
        
        # Choose the architecture
        if config_arc == 'arc_1':
            conv1 = _conv_layer('conv1', self.features, [3, 3, self.img_dims[-1], 10])
            conv2 = _conv_layer('conv2', conv1, [3,3,10,30])
            conv3 = _conv_layer('conv3', conv2, [3,3,30,60])
            flatten3 = flatten(conv3)
            fc4 = _fc_layer('fc4', flatten3, 160, dropout=True)
            fc5 = _fc_layer('fc5', fc4, 84, dropout=True)
            fc6 = _fc_layer('fc6', fc5, self.n_class, dropout=False)
            logits = fc6

            return logits
        
        if config_arc == 'arc_2':
            conv1 = _conv_layer('conv1', self.features, [3, 3, self.img_dims[-1], 30]) # in: 32x32x1 out: 16x16x30
            conv2 = _conv_layer('conv2', conv1, [3,3,30,60]) #in: 16x16x30 out: 8x8x60
            conv3 = _conv_layer('conv3', conv2, [3,3,60,120])#in: 8x8x60 out:4x4x120
            flatten3 = flatten(conv3) #1920
            fc4 = _fc_layer('fc4', flatten3, 840, dropout=True)
            fc5 = _fc_layer('fc5', fc4, 256, dropout=True)
            fc6 = _fc_layer('fc6', fc5, self.n_class, dropout=False)
            logits = fc6

            return logits

    def _loss(self):
        self.logits = self._network_architect(self.config.arc)
        self.predictions = tf.nn.softmax(self.logits)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        # Can use different loss function, here we use the mean of cross entropy
        loss = tf.reduce_mean(cross_entropy)
        self.loss = loss

    def _optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        self.optimize = optimizer.minimize(self.loss)

    def _evaluate(self, X, y):
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        n_samples = len(X)
        batch_size = self.config.batch_size
        total_accuracy = 0
        for offset in range(0, n_samples, batch_size):
            end = offset + batch_size
            X_batch, y_batch = X[offset:end], y[offset:end]
            acc = self.sess.run(accuracy, feed_dict={self.features: X_batch,
                                                     self.labels: y_batch,
                                                     self.keep_prob: 1.0}) # no dropout when evaluate
            total_accuracy += (acc*len(X_batch)/n_samples)
        return total_accuracy

    def evaluate_model(self, X_=None, y_=None, model_path=None):
        """
        User need to specify the X_valid/X_test and y_valid/y_test
        If not given the model_path, will load the first_tuning_model by default
        """
        if (X_ is None) or (y_ is None):
            X_ = self.test_features
            y_ = self.test_labels
            print('use the default test set in pickle file')
        assert(X_.shape[0] == y_.shape[0])
        self._start_sess()

        if model_path is None:
            model_path = self.best_model_save_path

        assert(tf.train.get_checkpoint_state(model_path) is True, model_path+ 'does not exist!')
        self.saver.restore(self.sess, model_path)
        acc = self._evaluate(X_,y_)
        
        print('Overall accuracy is {:.3f}%'.format(acc*100))

        self.test_all_labels_acc = []

        for i in range(self.n_class):
            indices = np.where(np.argmax(y_, 1) == i)[0]
            X_i = X_[indices] # extract the specific class
            y_i = y_[indices]
            acc_i = self._evaluate(X_i, y_i)
            self.test_all_labels_acc.append(acc_i)
            print('class {}, test_num {}, acc = {:.3f}%'.format(i, len(indices), acc_i*100))
        
        # Visualize
        plt.plot(range(len(self.test_all_labels_acc)), self.test_all_labels_acc, '-r')
        
        plt.show()
        self._reset_sess()

    def train(self, fine_tuning=False, best_model_save_path=None):
        '''
        Offer the fine_tuning
        '''
        self._start_sess()
        if fine_tuning:
            if best_model_save_path is None:
                model_path = self.best_model_save_path
            assert(tf.train.get_checkpoint_state(model_path) is True, model_path+ 'doest not exist!')
            self.saver.restore(self.sess, model_path)
        else:
            self.sess.run(self.init_variables) 
            
        EPOCHS = self.config.epochs
        batch_size = self.config.batch_size
        n_examples = len(self.train_features)

        self.best_epoch = -1 # Note the best model_path
        self.loss_batch = []
        self.train_acc_batch = []
        self.valid_acc_batch = []
        self.test_acc_batch = []


        print('Traing start..')
        for epoch in range(EPOCHS):
            start_time = time.time()

            # Very important schffle!
            self.train_features, self.train_labels = shuffle(self.train_features, self.train_labels)
            for offset in range(0, n_examples, batch_size):
                end = offset + batch_size
                X_batch, y_batch = self.train_features[offset:end], self.train_labels[offset:end]
                _, _loss = self.sess.run([self.optimize, self.loss], feed_dict={self.features: X_batch,
                                                                                self.labels: y_batch,
                                                                                self.keep_prob: 0.5})
            end_time = time.time()
            speed = int(n_examples // (end_time - start_time))
            self.loss_batch.append(_loss)
            train_acc = self._evaluate(self.train_features, self.train_labels)
            self.train_acc_batch.append(train_acc)
            valid_acc = self._evaluate(self.valid_features, self.valid_labels)
            self.valid_acc_batch.append(valid_acc)
            test_acc = self._evaluate(self.test_features, self.test_labels)
            self.test_acc_batch.append(test_acc)
            print('Epoch {}: loss= {:.2f}, train_acc= {:.3f}%, valid_acc= {:.3f}%, test_acc= {:.3f}%, speed= {:d} images/s'.format(epoch+1, _loss, train_acc*100, valid_acc*100, test_acc*100,speed))

            #[Note done the best tuning epoch]
            if (epoch+1) > EPOCHS//5:
                best_epoch = np.argmax(np.array(self.train_acc_batch) + np.array(self.valid_acc_batch)) #TBD: valid_acc should have more weighted?
                if best_epoch > self.best_epoch:
                    self.best_epoch = best_epoch # update!
                    self.saver.save(self.sess, self.best_model_save_path)
                    print('[Update] the bset model at epoch {:d}'.format(epoch+1))
        print('Best model at epoch {:d}, train_acc= {:.3f}%, valid_acc= {:.3f}%'.format(self.best_epoch+1, self.train_acc_batch[self.best_epoch]*100, self.valid_acc_batch[self.best_epoch]*100))
        writer = tf.summary.FileWriter('summary/'+self.config.arc, self.sess.graph)
        self._reset_sess()
        
    def predict(self, X_new, model_path=None ,top_k=1):
        '''
        predict the new dataset with the best model
        return softmax values with top k 
        '''
        self._start_sess()
        if model_path is None:
            model_path = self.best_model_save_path
        assert(tf.train.get_checkpoint_state(model_path) is True, model_path+ 'doest not exist!')
        self.saver.restore(self.sess, model_path)
        
        # Check the self.predictions -> no self.labels needed!
        pred = self.sess.run(self.predictions, feed_dict={self.features: X_new,
                                                          self.keep_prob: 1.0})
        
        result = self.sess.run(tf.nn.top_k(tf.constant(pred), k=top_k))
        return result
        
