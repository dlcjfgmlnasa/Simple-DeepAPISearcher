# train.py
from model.data_helper import PreDataProcessing
from model.deepAPI_model import Seq2Seq
from config import FLAGS
import tensorflow as tf
import os


def main(_):
    # pre processing
    p2d_processing = PreDataProcessing()
    p2d_processing.load_file_dir(FLAGS.train_dir)
    p2d_processing.make_data_set()
    encoder_size = p2d_processing.get_encoder_size()
    decoder_size = p2d_processing.get_decoder_size()
    encoder_vocab_size = p2d_processing.get_encoder_vocab_size()
    decoder_vocab_size = p2d_processing.get_decoder_vocab_size()

    p2d_processing.load_encoder_and_decoder_dic(FLAGS.dictionary_dir,
                                                FLAGS.encoder_dic_name,
                                                FLAGS.decoder_dic_name)

    batch_iter = p2d_processing.iter_batch(epochs=FLAGS.epochs, batch_size=FLAGS.batch_size)

    # seq2seq model
    seq2seq = Seq2Seq(encoder_size=encoder_size,
                      decoder_size=decoder_size,
                      encoder_vocab_size=encoder_vocab_size,
                      decoder_vocab_size=decoder_vocab_size,
                      encoder_layer_size=FLAGS.encoder_layer_size,
                      decoder_layer_size=FLAGS.decoder_layer_size,
                      RNN_type=FLAGS.RNN_type,
                      encoder_input_keep_prob=0.7,
                      encoder_output_keep_prob=1.0,
                      decoder_input_keep_prob=0.7,
                      decoder_output_keep_prob=1.0,
                      learning_rate=FLAGS.learning_rate,
                      hidden_size=FLAGS.hidden_layer_size)

    with tf.Session() as sess:
        # remove tensor board directory
        if os.path.exists(FLAGS.board_dir):
            for tensor_board_file in os.listdir(FLAGS.board_dir):
                tensor_board_file = os.path.join(FLAGS.board_dir, tensor_board_file)
                os.remove(tensor_board_file)

        # saver model
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(FLAGS.board_dir, graph=sess.graph)

        for i, batch in enumerate(batch_iter):
            enc_input, dec_input, tar_input = batch
            merged, cost, _ = seq2seq.train(sess, enc_input, dec_input, tar_input)
            writer.add_summary(merged, global_step=i)

            # print('encoder:{}, decoder:{}, target:{}'.format(enc_input, dec_input, tar_input))
            if i % 10 == 0 :
                print('i : {} cost : {}'.format(i, cost))

        # save model
        seq2seq.saver.save(sess, FLAGS.model_dir, latest_filename=FLAGS.model_name)

if __name__ == '__main__':
    tf.app.run()