# work.py

from model.deepAPI_model import Seq2Seq
from model.data_helper import PreDataProcessing
import tensorflow as tf
import os
from config import FLAGS


def main(_):
    print('wait some time')

    encoder_dic_dir = os.path.join(FLAGS.dictionary_dir, FLAGS.encoder_dic_name)
    decoder_dic_dir = os.path.join(FLAGS.dictionary_dir, FLAGS.decoder_dic_name)

    p2d_processing = PreDataProcessing()
    p2d_processing.load_file_dir(FLAGS.train_dir)
    p2d_processing.set_helper_setting(encoder_dic_dir, decoder_dic_dir)
    print('setting end!!')

    seq2seq = Seq2Seq(
        encoder_size=p2d_processing.get_encoder_size(),
        decoder_size=p2d_processing.get_decoder_size(),
        encoder_vocab_size=p2d_processing.get_encoder_vocab_size(),
        decoder_vocab_size=p2d_processing.get_decoder_vocab_size(),
        encoder_layer_size=FLAGS.encoder_layer_size,
        decoder_layer_size=FLAGS.decoder_layer_size,
        RNN_type=FLAGS.RNN_type,
        encoder_input_keep_prob=1.0,
        encoder_output_keep_prob=1.0,
        decoder_input_keep_prob=1.0,
        decoder_output_keep_prob=1.0,
        learning_rate=FLAGS.learning_rate,
        hidden_size=FLAGS.hidden_layer_size
    )

    decoder_dic_list = {value: key for key, value in p2d_processing.get_decoder_dic().items()}
    padding_list = p2d_processing.get_padding_list()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # model restore
        seq2seq.saver.restore(sess=sess, save_path=FLAGS.model_dir)

        while True:
            sentence = input('값을 입력하시오!!')
            encoder, decoder = p2d_processing.sentence_apply_padding_and_vector(sentence)
            output = seq2seq.predicate(session=sess,
                                       enc_input=encoder,
                                       dec_input=decoder)[0]
            api_list = []
            for i in output:
                if not decoder_dic_list[i] in padding_list:
                    api_list.append(decoder_dic_list[i])
            print(api_list)

if __name__ == '__main__':
    tf.app.run()