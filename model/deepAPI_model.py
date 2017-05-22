# chatbot_model.py

import tensorflow as tf
from tensorflow.contrib import rnn
from model.data_helper import MakeDictionary, AdvPreProcessing


class Seq2Seq(object):
    def __init__(self,
                 encoder_size,
                 decoder_size,
                 encoder_vocab_size,
                 decoder_vocab_size,
                 encoder_layer_size,
                 decoder_layer_size,
                 RNN_type='LSTM',
                 encoder_input_keep_prob=1.0,
                 encoder_output_keep_prob=1.0,
                 decoder_input_keep_prob=1.0,
                 decoder_output_keep_prob=1.0,
                 learning_rate=0.01,
                 hidden_size=128):

        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.encoder_layer_size = encoder_layer_size
        self.decoder_layer_size = decoder_layer_size
        self.encoder_input_keep_prob = encoder_input_keep_prob
        self.encoder_output_keep_prob = encoder_output_keep_prob
        self.decoder_input_keep_prob = decoder_input_keep_prob
        self.decoder_output_keep_prob = decoder_output_keep_prob
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size

        self.encoder_input = tf.placeholder(tf.float32, shape=(None, self.encoder_size, self.encoder_vocab_size))
        self.decoder_input = tf.placeholder(tf.float32, shape=(None, self.decoder_size, self.decoder_vocab_size))
        self.target_input = tf.placeholder(tf.int32, shape=(None, self.decoder_size))

        self.weight = tf.get_variable(shape=[self.hidden_size, self.decoder_vocab_size],
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      dtype=tf.float32,
                                      name='weight')
        self.bias = tf.get_variable(shape=[self.decoder_vocab_size],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    dtype=tf.float32,
                                    name='bias')

        self.logits = None
        self.cost = None
        self.train_op = None
        self.RNNCell = None
        self.outputs = None
        self.merged = None

        if RNN_type == 'LSTM':
            self.RNNCell = rnn.LSTMCell
        elif RNN_type == 'GRU':
            self.RNNCell = rnn.GRUCell
        else:
            raise Exception('not support {} RNN type'.format(RNN_type))

        self.build_model()
        self.saver = tf.train.Saver(tf.global_variables())

    def build_model(self):
        encoder_cell, decoder_cell = self.build_cells()

        with tf.variable_scope('encode'):
            outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_input, dtype=tf.float32)
            tf.summary.histogram('encoder_output', outputs)

        with tf.variable_scope('decode'):
            outputs, decoder_state = tf.nn.dynamic_rnn(decoder_cell, self.decoder_input,
                                                       initial_state=encoder_state, dtype=tf.float32)
            tf.summary.histogram('decoder_output', outputs)

        self.logits, self.cost, self.train_op = self.build_ops(outputs, self.target_input)
        self.outputs = tf.argmax(self.logits, 2)
        self.merged = tf.summary.merge_all()

    def build_cells(self):
        # encoder cell
        with tf.name_scope('encoder_cell') as scope:
            encoder_cell = rnn.MultiRNNCell([self.RNNCell(num_units=self.hidden_size)
                                             for _ in range(self.encoder_layer_size)])
            encoder_cell = rnn.DropoutWrapper(encoder_cell,
                                              input_keep_prob=self.encoder_input_keep_prob,
                                              output_keep_prob=self.encoder_output_keep_prob)

        # decoder cell
        with tf.name_scope('decoder_cell') as scope:
            decoder_cell = rnn.MultiRNNCell([self.RNNCell(num_units=self.hidden_size)
                                             for _ in range(self.decoder_layer_size)])
            decoder_cell = rnn.DropoutWrapper(decoder_cell,
                                              input_keep_prob=self.decoder_input_keep_prob,
                                              output_keep_prob=self.decoder_output_keep_prob)

        return encoder_cell, decoder_cell

    def build_ops(self, outputs, targets):
        time_steps = tf.shape(outputs)[1]

        outputs = tf.reshape(outputs, [-1, self.hidden_size])
        logits = tf.matmul(outputs, self.weight) + self.bias
        logits = tf.reshape(logits, [-1, time_steps, self.decoder_vocab_size])

        with tf.name_scope('cost') as scope:
            cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
            tf.summary.scalar('cost', cost)

        with tf.name_scope('train_op') as scope:
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        return logits, cost, train_op

    def train(self, session, enc_input, dec_input, tar_input):
        return session.run(
            [self.merged, self.cost, self.train_op],
            feed_dict={
                self.encoder_input: enc_input,
                self.decoder_input: dec_input,
                self.target_input: tar_input
            }
        )

    def test(self, session, enc_input, dec_input, tar_input):
        pass

    def predicate(self, session, enc_input, dec_input):
        return session.run(
            self.outputs,
            feed_dict={
                self.encoder_input: enc_input,
                self.decoder_input: dec_input
            }
        )


class AdvSeq2Seq(object):
    def __init__(self,
                 encoder_sequence_size,
                 decoder_sequence_size,
                 encoder_vocab_embedding_size,
                 decoder_vocab_embedding_size,
                 RNN_type='LSTM',
                 encoder_layer_size=3,
                 decoder_layer_size=3,
                 encoder_input_keep_prob=1.0,
                 encoder_output_keep_prob=1.0,
                 decoder_input_keep_prob=1.0,
                 decoder_output_keep_prob=1.0,
                 hidden_layer_size=128,
                 learning_rate=0.001):
        self.encoder_sequence_size = encoder_sequence_size
        self.decoder_sequence_size = decoder_sequence_size
        self.encoder_vocab_embedding_size = encoder_vocab_embedding_size
        self.decoder_vocab_embedding_size = decoder_vocab_embedding_size
        self.RNN_type = RNN_type
        self.encoder_layer_size = encoder_layer_size
        self.decoder_layer_size = decoder_layer_size
        self.encoder_input_keep_prob = encoder_input_keep_prob
        self.encoder_output_keep_prob = encoder_output_keep_prob
        self.decoder_input_keep_prob = decoder_input_keep_prob
        self.decoder_output_keep_prob = decoder_output_keep_prob
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate

        self.encoder_input = tf.placeholder(shape=[None, self.encoder_sequence_size],
                                            dtype=tf.int32, name='encoder_input')
        self.decoder_input = tf.placeholder(shape=[None, self.decoder_sequence_size],
                                            dtype=tf.int32, name='decoder_input')
        self.decoder_target = tf.placeholder(shape=[None, self.decoder_sequence_size],
                                             dtype=tf.int32, name='decoder_target')

        self.RNN = None
        self.logit = None
        self.train_op = None
        self.cost = None
        self.output = None
        self.merged = None

        if RNN_type == 'LSTM':
            self.RNN = rnn.LSTMCell
        elif RNN_type == 'GRU':
            self.RNN = rnn.GRUCell
        else:
            print('{} is not exist rnn type'.format(RNN_type))
            print('error !! I wish input rnn type is LSTM or GRU model!!!')
            return

        encoder_embedding = tf.Variable(tf.random_normal(
            [self.encoder_sequence_size, self.encoder_vocab_embedding_size], -1.0, 1.0,
            name='encoder_embedding',
            dtype=tf.float32
        ))

        decoder_embedding = tf.Variable(tf.random_normal(
            [self.decoder_sequence_size, self.decoder_vocab_embedding_size], -1.0, 1.0,
            name='decoder_embedding',
            dtype=tf.float32
        ))

        self.encoder_input_embedding = tf.nn.embedding_lookup(encoder_embedding, self.encoder_input,
                                                              name='encoder_input_embedding')
        self.decoder_input_embedding = tf.nn.embedding_lookup(decoder_embedding, self.decoder_input,
                                                              name='decoder_input_embedding')

        self.weight = tf.get_variable(
            name='weight',
            shape=[self.hidden_layer_size, self.decoder_vocab_embedding_size],
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=tf.float32
        )
        self.bias = tf.get_variable(
            name='bias',
            shape=[self.decoder_vocab_embedding_size],
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=tf.float32
        )

        self.__build_model()
        self.saver = tf.train.Saver(tf.global_variables())

    def __build_model(self):
        encoder_cell, decoder_cell = self.__build_rnn_cell()

        with tf.variable_scope('encoder_layer'):
            encoder_output, encoder_state = tf.nn.dynamic_rnn(
                cell=encoder_cell,
                inputs=self.encoder_input_embedding,
                dtype=tf.float32
            )
            tf.summary.histogram('encoder_output', encoder_output)
            del encoder_output

        with tf.variable_scope('decoder_layer'):
            output, decoder_state = tf.nn.dynamic_rnn(
                cell=decoder_cell,
                inputs=self.decoder_input_embedding,
                initial_state=encoder_state,
                dtype=tf.float32
            )
            tf.summary.histogram('decoder_layer', output)
            del decoder_state

        self.logit, self.cost, self.train_op = self.__build_ops(output)
        self.output = tf.arg_max(self.logit, 2)
        self.merged = tf.summary.merge_all()

    def __build_ops(self, output):
        time_step = tf.shape(output)[1]
        output = tf.reshape(output, [-1, self.hidden_layer_size])
        logits = tf.matmul(output, self.weight) + self.bias
        logits = tf.reshape(logits, [-1, time_step, self.decoder_vocab_embedding_size])

        with tf.name_scope('cost'):
            cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                                 labels=self.decoder_target))
            tf.summary.scalar('cost', cost)

        with tf.name_scope('train_op'):
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        return logits, cost, train_op

    def __build_rnn_cell(self):
        with tf.name_scope('encoder_cell'):
            encoder_cell = rnn.MultiRNNCell([self.RNN(num_units=self.hidden_layer_size)
                                             for _ in range(self.encoder_layer_size)])
            encoder_cell = rnn.DropoutWrapper(
                cell=encoder_cell,
                input_keep_prob=self.encoder_input_keep_prob,
                output_keep_prob=self.encoder_output_keep_prob
            )

        with tf.name_scope('decoder_cell'):
            decoder_cell = rnn.MultiRNNCell([self.RNN(num_units=self.hidden_layer_size)
                                             for _ in range(self.decoder_layer_size)])
            decoder_cell = rnn.DropoutWrapper(
                cell=decoder_cell,
                input_keep_prob=self.decoder_input_keep_prob,
                output_keep_prob=self.decoder_output_keep_prob
            )

        return encoder_cell, decoder_cell

    def train(self, session, encoder_input, decoder_input, decoder_target):
        return session.run(
            [self.merged, self.cost, self.train_op],
            feed_dict={
                self.encoder_input: encoder_input,
                self.decoder_input: decoder_input,
                self.decoder_target: decoder_target
            }
        )

    def predicate(self, session, encoder_input, decoder_input):
        return session.run(
            self.output,
            feed_dict={
                self.encoder_input: encoder_input,
                self.decoder_input: decoder_input
            }
        )

if __name__ == '__main__':
    pass