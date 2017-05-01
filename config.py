import tensorflow as tf

# directory
tf.flags.DEFINE_string('train_dic', './data/train', 'train directory')
tf.flags.DEFINE_string('board_dic', './board', 'tensorboard directory')
tf.flags.DEFINE_string('model_dic', './tmp/model.ckpt', 'deep API Searching directory')

tf.flags.DEFINE_integer('epochs', 50, 'epoch number')
tf.flags.DEFINE_integer('batch_size', 4, 'batch number (even!!)')
tf.flags.DEFINE_integer('learning_rate', 0.001, 'AdamOptimizer minimizer')
tf.flags.DEFINE_integer('hidden_layer_size', 300,'RNN hidden layer size')
tf.flags.DEFINE_integer('encoder_layer_size', 3, 'encoder layer size')
tf.flags.DEFINE_integer('decoder_layer_size', 3, 'decoder layer size')
FLAGS = tf.flags.FLAGS