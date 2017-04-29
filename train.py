# train.py
from model.data_helper import PreDataProcessing
from model.deepAPI_model import Seq2Seq
import tensorflow as tf
import os

# directory
tf.flags.DEFINE_string('train_dic', './data/train', 'train directory')
tf.flags.DEFINE_string('board_dic', './board', 'tensorboard directory')

tf.flags.DEFINE_integer('epochs', 50, 'epoch number')
tf.flags.DEFINE_integer('batch_size', 4, 'batch number (even!!)')
tf.flags.DEFINE_integer('learning_rate', 0.001, 'AdamOptimizer minimizer')
tf.flags.DEFINE_integer('hidden_layer_size', 300,'RNN hidden layer size')
tf.flags.DEFINE_integer('encoder_layer_size', 3, 'encoder layer size')
tf.flags.DEFINE_integer('decoder_layer_size', 3, 'decoder layer size')
FLAGS = tf.flags.FLAGS


def main(_):
    # pre processing
    p2d_processing = PreDataProcessing()
    p2d_processing.load_file_dir(FLAGS.train_dic)
    p2d_processing.make_data_set()
    encoder_size = p2d_processing.get_encoder_size()
    decoder_size = p2d_processing.get_decoder_size()
    encoder_vocab_size = p2d_processing.get_encoder_vocab_size()
    decoder_vocab_size = p2d_processing.get_decoder_vocab_size()
    batch_iter = p2d_processing.iter_batch(epochs=FLAGS.epochs, batch_size=FLAGS.batch_size)

    # seq2seq model
    seq2seq = Seq2Seq(encoder_size=encoder_size,
                      decoder_size=decoder_size,
                      encoder_vocab_size=encoder_vocab_size,
                      decoder_vocab_size=decoder_vocab_size,
                      encoder_layer_size=FLAGS.encoder_layer_size,
                      decoder_layer_size=FLAGS.decoder_layer_size,
                      RNN_type='LSTM',
                      encoder_input_keep_prob=0.7,
                      encoder_output_keep_prob=1.0,
                      decoder_input_keep_prob=0.7,
                      decoder_output_keep_prob=1.0,
                      learning_rate=FLAGS.learning_rate,
                      hidden_size=FLAGS.hidden_layer_size)

    with tf.Session() as sess:
        # remove tensor board directory
        if os.path.exists(FLAGS.board_dic):
            os.removedirs(FLAGS.board_dic)

        writer = tf.summary.FileWriter(FLAGS.board_dic, graph=sess.graph)
        sess.run(tf.global_variables_initializer())
        for i, batch in enumerate(batch_iter):
            enc_input, dec_input, tar_input = batch
            merged, cost, _ = seq2seq.train(sess, enc_input, dec_input, tar_input)
            writer.add_summary(merged, global_step=i)
            if i % 10 == 0:
                print('cost : ', cost)


if __name__ == '__main__':
    tf.app.run()