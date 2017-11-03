import tensorflow as tf

# directory
tf.flags.DEFINE_string('train_dir', './data/train', 'train directory')
tf.flags.DEFINE_string('content_file_name', 'contents', 'contents file list')
tf.flags.DEFINE_string('label_file_name', 'label', 'label file list')

tf.flags.DEFINE_string('board_dir', './board', 'tensor board directory')
tf.flags.DEFINE_string('model_dir', './tmp/model/', 'deep API Searching directory')
tf.flags.DEFINE_string('model_name', 'model.ckpt', 'model name')

tf.flags.DEFINE_string('dictionary_dir', './tmp/dictionary', 'encoder and decoder directory')
tf.flags.DEFINE_string('encoder_dic_name', 'encoder.dic', 'encoder dictionary name')
tf.flags.DEFINE_string('decoder_dic_name', 'decoder.dic', 'decoder dictionary name')

tf.flags.DEFINE_integer('epochs', 2000, 'epoch number')
tf.flags.DEFINE_integer('batch_size', 20, 'batch number (even!!)')
tf.flags.DEFINE_integer('learning_rate', 0.001, 'AdamOptimizer minimizer')
tf.flags.DEFINE_integer('hidden_layer_size', 300, 'RNN hidden layer size')
tf.flags.DEFINE_integer('encode_sequence_length', 50, 'encode rnn sequence length size')
tf.flags.DEFINE_integer('decode_sequence_length', 50, 'decode rnn sequence length size')
tf.flags.DEFINE_integer('encoder_layer_size', 1, 'encoder layer size')
tf.flags.DEFINE_integer('decoder_layer_size', 1, 'decoder layer size')
tf.flags.DEFINE_string('RNN_type', 'GRU', 'rnn type')
tf.flags.DEFINE_string('batch_num_threads', 4, 'batch num threads')
FLAGS = tf.flags.FLAGS