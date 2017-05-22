import tensorflow as tf
import os
from model.data_helper import MakeDictionary, AdvPreProcessing
from model.deepAPI_model import AdvSeq2Seq
from config import FLAGS

if __name__ == '__main__':
    make_dictionary = MakeDictionary(
        encoder_data_dir=os.path.join(FLAGS.train_dir, FLAGS.content_file_name),
        decoder_data_dir=os.path.join(FLAGS.train_dir, FLAGS.label_file_name),
        encoder_dic_dir=os.path.join(FLAGS.dictionary_dir, FLAGS.encoder_dic_name),
        decoder_dic_dir=os.path.join(FLAGS.dictionary_dir, FLAGS.decoder_dic_name)
    )
    encoder_dic, decoder_dic = make_dictionary.load_dic()
    encoder_max_len = len(encoder_dic)
    decoder_max_len = len(decoder_dic)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    print(decoder_dic)

    adv_p2processing = AdvPreProcessing(
        session=session,
        enc_dic=encoder_dic,
        dec_dic=decoder_dic,
        enc_sequence_length=FLAGS.encode_sequence_length,
        dec_sequence_length=FLAGS.decode_sequence_length
    )
    enc_padding_list, dec_padding_list = adv_p2processing.apply_sentence_to_padding('I am a boy')

    seq2seq = AdvSeq2Seq(
        encoder_sequence_size=FLAGS.encode_sequence_length,
        decoder_sequence_size=FLAGS.decode_sequence_length,
        encoder_vocab_embedding_size=encoder_max_len,
        decoder_vocab_embedding_size=decoder_max_len,
        RNN_type='LSTM',
        encoder_layer_size=1,
        decoder_layer_size=1,
        encoder_input_keep_prob=1.0,
        encoder_output_keep_prob=1.0,
        decoder_input_keep_prob=1.0,
        decoder_output_keep_prob=1.0,
        hidden_layer_size=128,
        learning_rate=0.001
    )

    seq2seq.saver.restore(sess=session, save_path=FLAGS.model_dir)
