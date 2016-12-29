class Model(object):
    """Abstracts a Tensorflow graph for a learning task.

    Various Model classes can be used as usual abstractions to encapsulate
    Tensorflow computational graphs.
    """

    def load_data(self):
        """Loads the data (from disk or by function call) and stores it in memory.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_placeholders(self):
        """Adds placeholder variables to Tensorflow computational graph.

        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are
        used as inputs by the rest of the model building code and will be fed
        data during training.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, input_batch, label_batch):
        """Creates the feed_dict for training the given step.

        A feed_dict takes the form of:

            feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
            }

        Args:
          input_batch: A batch of input data.
          label_batch: A batch of label data.
        Returns:
          feed_dict: The feed dictionary mapping from placeholders to values.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_model(self, input_data):
        """Implement a graph that transforms input_data into predictions.

        The core transformation for this model which transforms a batch of
        input data into a batch of predictions.

        Args:
          input_data: A tensor of shape (batch_size, n_features).
        Returns:
          out: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        """Adds ops for loss to the computational graph.

        Args:
          pred: A tensor of shape (batch_size, n_classes)
        Returns:
          loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable
        variables.  The Op returned by this function is what must be passed to
        the `sess.run()` call to cause the model to train.

        Args:
          loss: Loss tensor, from cross_entropy_loss.
        Returns:
          train_op: The Op for training.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def run_epoch(self, sess, input_data, input_labels):
        """Runs an epoch of training.

        Args:
          sess: tf.Session() object
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
          average_loss: scalar. Average minibatch loss of model on epoch.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def fit(self, sess, input_data, input_labels):
        """Fit model on provided data.

        Args:
          sess: tf.Session()
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
          losses: list of loss per epoch
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    # def predict(self, sess, input_data, input_labels=None):
    #     """Make predictions from the provided model.
    #
    #     Args:
    #       sess: tf.Session()
    #       input_data: np.ndarray of shape (n_samples, n_features)
    #       input_labels: np.ndarray of shape (n_samples, n_classes)
    #     Returns:
    #       average_loss: Average loss of model.
    #       predictions: Predictions of model on input_data
    #     """
    #     raise NotImplementedError("Each Model must re-implement this method.")
