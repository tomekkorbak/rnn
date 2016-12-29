import tensorflow as tf
import numpy as np

from model import Model
from xor import generate_xor_array
from utils import data_iterator


class Config(object):
    """Holds model hyperparameters and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    num_samples_dev = 800
    num_samples_test = 200
    num_steps = 100
    num_neurons = 5
    num_layers = 1
    num_classes = 2
    dropout = 0.5
    lr = 1e-4
    batch_size = 100


class XorModel(Model):

    def load_data(self):
        self.data = generate_xor_array(1000, 30)
        self.labels = np.ones([1000, self.config.num_classes], dtype=int)

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(
            tf.float32, shape=[self.config.batch_size, 30, 1], name='input')
        self.labels_placeholder = tf.placeholder(
            tf.float32, shape=[self.config.batch_size, self.config.num_classes], name='target')

    def create_feed_dict(self, input_batch, label_batch):
        feed_dict = {
            self.input_placeholder: input_batch,
            self.labels_placeholder: label_batch,
        }
        return feed_dict


class SimpleRNN(XorModel):

    def add_model(self, input_data):
        with tf.variable_scope("layer1", reuse=None):
            basic_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.num_neurons)
            cell_with_dropout = tf.nn.rnn_cell.DropoutWrapper(
                cell=basic_cell,
                output_keep_prob=self.config.dropout
            )
            network = tf.nn.rnn_cell.MultiRNNCell(
                [cell_with_dropout] * self.config.num_layers
            )
            outputs, _ = tf.nn.dynamic_rnn(
                network,
                input_data,
                dtype=tf.float32
            )

            # Get outputs' last element and size
            # outputs = tf.transpose(outputs, [1, 0])
            last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

            # Initialize additional trainable variables
            weight = tf.Variable(
                tf.truncated_normal([self.config.num_neurons, 1],
                                    stddev=0.1)
            )
            bias = tf.Variable(tf.constant(0.1, shape=[1]))
            y = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return y

    def add_loss_op(self, prediction):
        return -1 * tf.reduce_sum(self.labels_placeholder * tf.log(prediction))

    def add_training_op(self, loss):
        opt = tf.train.GradientDescentOptimizer(learning_rate=self.config.lr)
        return opt.minimize(loss)

    def fit(self, session, input_data, input_labels):
        for epoch in range(10):
            loss = self.run_epoch(session, input_data, input_labels)
            print 'Epoch {:2d} loss {:3.1f}%'.format(epoch + 1, 100 * loss)

    def run_epoch(self, sess, input_data, input_labels):
        average_loss = 0
        for step, (input_batch, label_batch) in enumerate(
                data_iterator(input_data, input_labels,
                              batch_size=self.config.batch_size,
                              label_size=self.config.num_classes,
                              shuffle=True)):
            feed_dict = self.create_feed_dict(input_batch, label_batch)

            _, loss_value = sess.run([self.train_op, self.loss],
                                     feed_dict=feed_dict)
            average_loss += loss_value

        average_loss /= step
        return average_loss

    def __init__(self, config):
        """Initialize the model

        Args:
          config: A model configuration object of type Config
        """
        self.config = config
        # Generate placeholders for the images and labels.
        self.load_data()
        self.add_placeholders()
        self.pred = self.add_model(self.input_placeholder)
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
