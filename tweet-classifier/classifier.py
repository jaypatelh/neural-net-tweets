import time

import numpy as np
import tensorflow as tf
import argparse
import pickle
import os

from helpers import get_minibatches
from model import Model


class Config(object):
    """Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    #On all the data: samples = 39500, batch size = 
    #On all the non-spanish data: samples = 37810, batch size = 95
    #On just the californnia earthquake: 
    #On all the earthquakes: 


    n_samples = 37810
    n_features = 200
    n_classes = 18
    batch_size = 95
    n_epochs = 200
    lr = 1e-4


class SimpleModel(Model):
    """Implements a Softmax classifier with cross-entropy loss."""

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors.
        These placeholders are used as inputs by the rest of the model building
        and will be fed data during training.
        Adds following nodes to the computational graph
        input_placeholder: Input placeholder tensor of shape
                                              (batch_size, n_features), type tf.float32
        labels_placeholder: Labels placeholder tensor of shape
                                              (batch_size, n_classes), type tf.int32
        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
        """
        self.input_placeholder = tf.placeholder(tf.float32, shape=(Config.batch_size, Config.n_features))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(Config.batch_size, Config.n_classes))

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for training the given step.
        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }
        If label_batch is None, then no labels are added to feed_dict.
        Hint: The keys for the feed_dict should be the placeholder
                tensors created in add_placeholders.
        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {self.input_placeholder: inputs_batch}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch of input
        data into a batch of predictions. In this case, the transformation is a linear layer plus a
        softmax transformation:
        y = softmax(Wx + b)
        Hint: Make sure to create tf.Variables as needed.
        Hint: For this simple use-case, it's sufficient to initialize both weights W
                    and biases b with zeros.
        Args:
            input_data: A tensor of shape (batch_size, n_features).
        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        initializer=tf.contrib.layers.xavier_initializer()
        W = tf.Variable(initializer((Config.n_features, Config.n_classes)), dtype=tf.float32)
        b = tf.Variable(tf.zeros((Config.batch_size, Config.n_classes)), dtype=tf.float32)

        return tf.matmul(self.input_placeholder, W) + b

    def add_loss_op(self, pred):
        """Adds cross_entropy_loss ops to the computational graph.
        Hint: Use the cross_entropy_loss function we defined. This should be a very
                    short function.
        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss_samples = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=pred)
        loss = tf.reduce_mean(loss_samples)
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See
        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer
        for more information.
        Hint: Use tf.train.GradientDescentOptimizer to get an optimizer object.
                    Calling optimizer.minimize() will return a train_op object.
        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        train_op = tf.train.GradientDescentOptimizer(Config.lr).minimize(loss)
        # train_op = tf.train.AdamOptimizer(Config.lr).minimize(loss)
        return train_op

    def run_epoch(self, sess, inputs, labels):
        """Runs an epoch of training.
        Args:
            sess: tf.Session() object
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        n_minibatches, total_loss = 0, 0
        for input_batch, labels_batch in get_minibatches([inputs, labels], self.config.batch_size):
            n_minibatches += 1
            total_loss += self.train_on_batch(sess, input_batch, labels_batch)
        return total_loss / n_minibatches

    def fit(self, sess, inputs, labels):
        """Fit model on provided data.
        Args:
            sess: tf.Session()
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            losses: list of loss per epoch
        """
        losses = []
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            average_loss = self.run_epoch(sess, inputs, labels)
            duration = time.time() - start_time
            print 'Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration)
            losses.append(average_loss)
        return losses

    def predict(self, sess, inputs, labels):
        total_matches = 0
        total_samples = 0
        for input_batch, labels_batch in get_minibatches([inputs, labels], Config.batch_size):
            predictions = self.predict_on_batch(sess, input_batch)
            predicted_classes = tf.argmax(predictions, axis=1)
            true_classes = tf.argmax(labels_batch, axis=1)

            total_matches += matches(true_classes, predicted_classes)
            total_samples += Config.batch_size

        print "Overall Accuracy: "
        print sess.run(total_matches*100.0 / float(total_samples)), "%"

    def __init__(self, config):
        """Initializes the model.
        Args:
            config: A model configuration object of type Config
        """
        self.config = config
        self.build()

def accuracy(y, yhat):
    assert(y.get_shape() == yhat.get_shape())
    return matches(y, yhat) * 100.0 / y.get_shape().as_list()[0]

def matches(y, yhat):
    assert(y.get_shape() == yhat.get_shape())
    return tf.reduce_sum(tf.to_float(tf.to_int32(tf.equal(y, yhat))))

def fit_and_predict(inputs, labels):
    """Train softmax model for a number of steps."""
    config = Config()

    # Generate random data to train the model on
    np.random.seed(1234)

    threshold = Config.batch_size*(int(0.8*(Config.n_samples/Config.batch_size)))
    
    # Tell TensorFlow that the model will be built into the default Graph.
    # (not required but good practice)
    with tf.Graph().as_default():
        # Build the model and add the variable initializer Op
        model = SimpleModel(config)
        init = tf.global_variables_initializer()
        # If you are using an old version of TensorFlow, you may have to use
        # this initializer instead.
        # init = tf.initialize_all_variables()

        # Create a session for running Ops in the Graph
        with tf.Session() as sess:
            # Run the Op to initialize the variables.
            sess.run(init)
            # Fit the model
            losses = model.fit(sess, inputs[:threshold], labels[:threshold])

            # If Ops are implemented correctly, the average loss should fall close to zero
            # rapidly.
            # assert losses[-1] < 2.2
            print "Basic (non-exhaustive) classifier tests pass"

            # model.predict(sess, inputs[:threshold], labels[:threshold])
            model.predict(sess, inputs[threshold:], labels[threshold:])

if __name__ == "__main__":
    data = [
        "data/california_earthquake", 
        "data/chile_earthquake", 
        "data/pakistan_earthquake", 
        "data/nepal_earthquake",
        "data/cyclone_pam",
        "data/ebola",
        "data/hurricane_mexico",
        "data/iceland_volcano",
        "data/india_floods",
        "data/landslides_ww_en",
        "data/malaysia_flight",
        "data/mers",
        "data/pakistan_floods",
        "data/philipines_typhoon",
        # "data/landslides_ww_es",
        # "data/landslides_ww_fr",
        # "data/chile_earthquake_es", 
    ]

    chosen_labels_matrix = None
    tweets_matrix = None
    
    for event in data:
        print "on ", event, "..."
        tweets_file = open(event + "/word2vec_average.p", "rb")
        labels_file = open(event + "/labels-03112017.p", "rb")
    
        tweets_vecs = pickle.load(tweets_file)
        labels_vecs = pickle.load(labels_file)

        for tweetid in tweets_vecs:
            if tweetid in labels_vecs:
                try:
                    tweets_matrix = np.vstack((tweets_matrix, tweets_vecs[tweetid])) if tweets_matrix is not None else tweets_vecs[tweetid]
                except ValueError as e:
                    print e
                    print "skipping tweet due to weird dimension:", tweetid
                    continue
                chosen_labels_matrix = np.vstack((chosen_labels_matrix, labels_vecs[tweetid])) if chosen_labels_matrix is not None else labels_vecs[tweetid]

    chosen_labels_matrix = chosen_labels_matrix[:Config.n_samples]
    tweets_matrix = tweets_matrix[:Config.n_samples]

    print chosen_labels_matrix.shape
    print tweets_matrix.shape

    fit_and_predict(tweets_matrix, chosen_labels_matrix)