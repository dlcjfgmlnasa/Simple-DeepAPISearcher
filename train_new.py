import tensorflow as tf
import os
import numpy as np
from model.data_helper import MakeDictionary, AdvPreProcessing
from model.deepAPI_model import AdvSeq2Seq
from config import FLAGS

if __name__ == '__main__':
    """
    advance train.py
    """
    sess = tf.Session()
    make_dic = MakeDictionary(
        encoder_data_dir=os.path.join(FLAGS.train_dir, FLAGS.content_file_name),
        decoder_data_dir=os.path.join(FLAGS.train_dir, FLAGS.label_file_name),
        encoder_dic_dir=os.path.join(FLAGS.dictionary_dir, FLAGS.encoder_dic_name),
        decoder_dic_dir=os.path.join(FLAGS.dictionary_dir, FLAGS.decoder_dic_name)
    )

    # load dictionary
    encoder_dic, decoder_dic = make_dic.load_dic()
    encoder_max_len = len(encoder_dic)
    decoder_max_len = len(decoder_dic)

    total_data_set_count = 0
    data_set_dir = os.path.join(FLAGS.train_dir, FLAGS.content_file_name)
    for file_name in os.listdir(data_set_dir):
        file_name = os.path.join(data_set_dir, file_name)
        total_data_set_count += sum(1 for _ in open(file_name).readlines())

    # load data set and make batch processing
    adv_iter = AdvPreProcessing(
        session=sess,
        enc_dic=encoder_dic,
        dec_dic=decoder_dic,
        enc_sequence_length=FLAGS.encode_sequence_length,
        dec_sequence_length=FLAGS.decode_sequence_length,
        contents_file_dir=os.path.join(FLAGS.train_dir, FLAGS.content_file_name),
        label_file_dir=os.path.join(FLAGS.train_dir, FLAGS.label_file_name),
        total_data_len=total_data_set_count,
        batch_size=FLAGS.batch_size
    )

    # setting seq2seq Algorithm : AdvSeq2Seq
    seq2seq = AdvSeq2Seq(
        encoder_sequence_size=FLAGS.encode_sequence_length,
        decoder_sequence_size=FLAGS.decode_sequence_length,
        encoder_vocab_embedding_size=encoder_max_len,
        decoder_vocab_embedding_size=decoder_max_len,
        encoder_layer_size=1,
        decoder_layer_size=1,
        RNN_type=FLAGS.RNN_type,
        encoder_input_keep_prob=0.8,
        encoder_output_keep_prob=1.0,
        decoder_input_keep_prob=0.8,
        decoder_output_keep_prob=1.0,
        hidden_layer_size=FLAGS.hidden_layer_size,
        learning_rate=FLAGS.learning_rate
    )

    # save board for tensor board
    if os.path.exists(FLAGS.board_dir):
        for board_filename in os.listdir(FLAGS.board_dir):
            board_filename = os.path.join(FLAGS.board_dir, board_filename)
            os.remove(board_filename)

    # saver model
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)
    writer = tf.summary.FileWriter(FLAGS.board_dir, graph=sess.graph)

    i = 0
    for _ in range(FLAGS.epochs):
        for _ in range(int(total_data_set_count / FLAGS.batch_size)):
            enc_input, dec_input, dec_target = next(adv_iter)
            merged, cost, train_op = seq2seq.train(sess, enc_input, dec_input, dec_target)
            writer.add_summary(merged, global_step=i)
            i = i+1

            if i % 10 == 0:
                print('i : {} cost : {}'.format(i, cost))

    # save!!
    seq2seq.saver.save(sess, FLAGS.model_dir, latest_filename=FLAGS.model_name)
    coord.request_stop()
    coord.join(thread)
    print('end train!!!')