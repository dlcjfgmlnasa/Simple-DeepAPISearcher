import tensorflow as tf

# directory
tf.flags.DEFINE_string('train_dir', './data/train', 'train directory')
tf.flags.DEFINE_string('board_dir', './board', 'tensorboard directory')
tf.flags.DEFINE_string('model_dir', './tmp/model/', 'deep API Searching directory')
tf.flags.DEFINE_string('model_name', 'model.ckpt', 'model name')
tf.flags.DEFINE_string('dictionary_dir', './tmp/dictionary', 'encoder and decoder directory')
tf.flags.DEFINE_string('encoder_dic_name', 'encoder.dic', 'encoder dictionary name')
tf.flags.DEFINE_string('decoder_dic_name', 'decoder.dic', 'decoder dictionary name')

tf.flags.DEFINE_integer('epochs', 50, 'epoch number')
tf.flags.DEFINE_integer('batch_size', 4, 'batch number (even!!)')
tf.flags.DEFINE_integer('learning_rate', 0.001, 'AdamOptimizer minimizer')
tf.flags.DEFINE_integer('hidden_layer_size', 300, 'RNN hidden layer size')
tf.flags.DEFINE_integer('encoder_layer_size', 3, 'encoder layer size')
tf.flags.DEFINE_integer('decoder_layer_size', 3, 'decoder layer size')
tf.flags.DEFINE_string('RNN_type', 'LSTM', 'rnn type')
FLAGS = tf.flags.FLAGS