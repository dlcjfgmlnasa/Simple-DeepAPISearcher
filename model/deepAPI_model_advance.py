import tensorflow as tf
from model.data_helper import PreDataProcessing
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple

class abvance_Seq2Seq(object):
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
        self.decoder_target = tf.placeholder(tf.int32, shape=(None, self.decoder_size))

        self.weight = tf.get_variable(
            shape=(self.hidden_size, self.decoder_vocab_size),
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=tf.float32,
            name='weight'
        )

        self.bias = tf.get_variable(
            shape=[self.decoder_vocab_size],
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=tf.float32,
            name='bias'
        )

        """
        양방향 RNN

        예측을 위해 표준 RNN을 사용할 때 "과거"만 고려합니다.
        특정 작업의 경우 이는 의미가 있지만 (예 : 다음 단어 예측), 일부 작업의 경우 과거와 미래를 모두 고려하는 것이 유용합니다.
        품사 태그 지정과 같이 문장의 각 단어에 태그를 지정하려는 태깅 작업을 생각해보십시오.
        여기서 우리는 이미 단어의 전체 순서를 알고 있으며, 각 단어에 대해 예측할 때 왼쪽 (과거) 단어뿐 아니라 오른쪽 (미래) 단어까지 고려해야합니다.
        양방향 RNN은 정확히 동일합니다. 양방향 RNN은 두 개의 RNN의 조합입니다. 
        하나는 "왼쪽에서 오른쪽으로"앞으로 가고 다른 하나는 "오른쪽에서 왼쪽으로"뒤로 실행됩니다. 
        이들은 태깅 작업에 일반적으로 사용되며,
        
        표준 RNN처럼 Tensorflow에는 양방향 RNN의 정적 및 동적 버전이 있습니다.
        이 글을 쓰는 시점에서, bidirectional_dynamic_rnn아직 문서화되지 않았지만 정적보다 선호 bidirectional_rnn됩니다.
        
        양방향 RNN 기능의 주요 차이점은 순방향 및 역방향 RNN에 대해 별도의 셀 인수를 사용하고 순방향 및 역방향 RNN에 대해 별도의 출력 및 상태를 반환한다는 것입니다.
        """

        cell1 = tf.contrib.rnn.LSTMCell(num_units=self.hidden_size / 2)
        cell2 = tf.contrib.rnn.LSTMCell(num_units=self.hidden_size / 2)

        with tf.variable_scope('test') as scope:
            (encoder_fw_output, encoder_bw_output), (encoder_fw_final_state, encoder_bw_final_state) = (
            tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell1,
                cell_bw=cell2,
                inputs=self.encoder_input,
                dtype=tf.float32

            ))
        print(encoder_fw_output)
        print(encoder_bw_output)
        print(encoder_fw_final_state)
        print(encoder_bw_final_state)

        predataprocessing = PreDataProcessing()
        predataprocessing.load_file_dir('../data/train')
        predataprocessing.make_data_set()
        batchs = predataprocessing.iter_batch(epochs=100, batch_size=4)
        batch = next(batchs)
        enc_input, dec_input, tar_input = batch

        # Have to concatenate forward and backward outputs and state. In this case we will not discard outputs,
        # they would be used for attention.

        encoder_outputs = tf.concat((encoder_fw_output, encoder_bw_output), 2)
        # print(sess.run(encoder_outputs, feed_dict={self.encoder_input: enc_input}))
        # print(sess.run(tf.shape(encoder_fw_output), feed_dict={self.encoder_input: enc_input}))
        # print(sess.run(tf.shape(encoder_outputs), feed_dict={self.encoder_input: enc_input}))

        encoder_final_state_c = tf.concat(
            (encoder_fw_final_state.c, encoder_bw_final_state.c), 1
        )

        encoder_final_state_h = tf.concat(
            (encoder_fw_final_state.h, encoder_bw_final_state.h), 1
        )

        encoder_final_state = tf.contrib.rnn.LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )

        decoder_cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_size)
        '''
        tf.nn.dynamic_rnn allows for easy RNN construction, but is limited.
        For example, a nice way to increase robustness of the model is to feed as decoder inputs tokens 
        that it previously generated,  instead of shifted true sequence.
        
        tf.nn.dynamic_rnn은 쉬운 RNN 구성을 허용하지만 제한적입니다.
        예를 들어, 모델의 견고성을 높이는 좋은 방법은 디코더 입력 토큰으로 공급하는 것입니다
        그것은 이전에 생성 된 것이고, 진정한 순서가 바뀌 었습니다.
        '''

        self.outputs, decoder_state = tf.nn.dynamic_rnn(
            cell=decoder_cell,
            inputs=self.decoder_input,
            initial_state=encoder_final_state,
            dtype=tf.float32
        )

        time_steps = tf.shape(self.outputs)[1]

        outputs = tf.reshape(self.outputs, [-1, self.hidden_size])
        logits = tf.matmul(outputs, self.weight) + self.bias
        logits = tf.reshape(logits, [-1, time_steps, self.decoder_vocab_size])
        self.outputs = tf.arg_max(self.outputs, 2)

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_target, logits=logits))
        self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.cost)

    def train(self, session, enc_input, dec_input, tar_input):
        return session.run(
            [self.cost, self.train_op],
            feed_dict={
                self.encoder_input: enc_input,
                self.decoder_input: dec_input,
                self.decoder_target: tar_input
            }
        )

    def predicate(self, session, enc_input, dec_input):
        return session.run(
            [self.outputs],
            feed_dict={
                self.encoder_input: enc_input,
                self.decoder_input: dec_input
            }
        )[0]

if __name__ == '__main__':
    tf.reset_default_graph()

    predataprocessing = PreDataProcessing()
    predataprocessing.load_file_dir('../data/train')
    predataprocessing.make_data_set()
    batchs = predataprocessing.iter_batch(epochs=100, batch_size=4)

    encoder_size = predataprocessing.get_encoder_size()
    decoder_size = predataprocessing.get_decoder_size()
    encoder_vocab_size = predataprocessing.get_encoder_vocab_size()
    decoder_vocab_size = predataprocessing.get_decoder_vocab_size()

    seq2seq = abvance_Seq2Seq(
        encoder_size=encoder_size,
        decoder_size=decoder_size,
        encoder_vocab_size=encoder_vocab_size,
        decoder_vocab_size=decoder_vocab_size,
        encoder_layer_size=3,
        decoder_layer_size=3
    )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for batch in batchs:
            enc_input, dec_input, tar_input = batch
            cost, train = seq2seq.train(sess, enc_input, dec_input, tar_input)
            print(cost)

        decoder_dic = {value: key for key, value in predataprocessing.get_decoder_dic().items()}
        print(decoder_dic)
        padding_list = predataprocessing.get_padding_list()
        print(padding_list)

        while True:
            sentence = input('input : ')
            if sentence.strip() == 'q':
                break
            enc_input, dec_input = predataprocessing.sentence_apply_padding_and_vector(sentence)
            outputs = seq2seq.predicate(sess, enc_input, dec_input)[0]
            result=[]
            for char in outputs:
                try:
                    temp = decoder_dic[char]
                    print(temp)
                except Exception as ex:
                    pass

