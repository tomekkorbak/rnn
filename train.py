import tensorflow as tf

from rnn import Config, SimpleRNN


def train_simple_rnn_model():
    config = Config()
    with tf.Graph().as_default():
        model = SimpleRNN(config)
        session = tf.Session()
        initalizer = tf.global_variables_initializer()
        session.run(initalizer)
        losses = model.fit(session, model.data, model.labels)
        print losses

if __name__ == "__main__":
    train_simple_rnn_model()