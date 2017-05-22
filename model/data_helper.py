# data_helper.py
import os
import nltk
import re
import copy
import pickle
import numpy as np
import tensorflow as tf
import itertools
import collections
import operator
from config import FLAGS


class PreDataProcessing(object):
    """
    data pre
    """
    def __init__(self):
        self.PAD = '_PAD_'
        self.EOS = '_EOS_'
        self.GO = '_GO_'
        self.UNK = '_UNK_'
        self.total_sentence_number = 0
        self.total_data_dic = {}
        self.encoder_data_dic = {}
        self.decoder_data_dic = {}
        self.target_data_dic = {}
        self.encoder_max_number = 0
        self.decoder_max_number = 0
        self.max_encoder_vocab_number = 0
        self.max_decoder_vocab_number = 0

        self.encoder_dic_dir = None
        self.decoder_dic_dir = None

        self.encoder_applied_padding_and_vector = []
        self.decoder_applied_padding_and_vector = []
        self.target_applied_padding_and_vector = []

    def get_padding_list(self):
        return [self.PAD, self.EOS, self.GO, self.PAD]

    def load_file_dir(self, file_dir):
        total_data_set = []
        for filename in os.listdir(file_dir):
            filename = os.path.join(file_dir, filename)
            total_data_set.extend([line for line in open(filename, 'r').readlines()])
        self.__split_data_set(total_data_set)

    def make_data_set(self):
        encoder_sentence_list = self.total_data_dic['q']
        decoder_sentence_list = self.total_data_dic['a']
        self.total_sentence_number = len(encoder_sentence_list)

        encoder_sentence_list = self.__encoder_tokenizer(encoder_sentence_list)
        decoder_sentence_list = self.__decoder_tokenizer(decoder_sentence_list)

        self.__make_dic(encoder_sentence_list, decoder_sentence_list)
        self.encoder_max_number = max([len(sentence) for sentence in encoder_sentence_list])
        self.decoder_max_number = max([len(sentence) for sentence in decoder_sentence_list])
        encoder_sentence_padding = self.__make_encoder_padding(encoder_sentence_list)
        decoder_sentence_padding = self.__make_decoder_padding(decoder_sentence_list)
        target_sentence_padding = self.__make_target_padding(decoder_sentence_list)
        self.encoder_applied_padding_and_vector = self.__apply_one_hot_vector(encoder_sentence_padding,
                                                                              self.encoder_data_dic)
        self.decoder_applied_padding_and_vector = self.__apply_one_hot_vector(decoder_sentence_padding,
                                                                              self.decoder_data_dic)
        self.target_applied_padding_and_vector = self.__apply_one_hot_vector(target_sentence_padding,
                                                                             self.decoder_data_dic, target_mode=True)

    def load_encoder_and_decoder_dic(self, dictionary_dic, encoder_name, decoder_name):
        if not os.path.exists(dictionary_dic):
            os.makedirs(dictionary_dic)
        encoder_dic_dir = os.path.join(dictionary_dic, encoder_name)
        decoder_dic_dir = os.path.join(dictionary_dic, decoder_name)

        encoder_output = open(encoder_dic_dir, 'wb')
        pickle.dump(self.encoder_data_dic, encoder_output)
        encoder_output.close()

        decoder_output = open(decoder_dic_dir, 'wb')
        pickle.dump(self.decoder_data_dic, decoder_output)
        decoder_output.close()

    def set_helper_setting(self, encoder_dic_dir, decoder_dic_dir):
        # load encoder, decoder dictionary
        self.encoder_dic_dir = encoder_dic_dir
        self.decoder_dic_dir = decoder_dic_dir
        encoder_file = open(self.encoder_dic_dir, 'rb')
        decoder_file = open(self.decoder_dic_dir, 'rb')

        # load encoder, decoder dic
        self.encoder_data_dic = pickle.load(encoder_file)
        self.decoder_data_dic = pickle.load(decoder_file)
        encoder_file.close()
        decoder_file.close()

        encoder_sentence_list = self.total_data_dic['q']
        decoder_sentence_list = self.total_data_dic['a']
        self.total_sentence_number = len(encoder_sentence_list)
        encoder_sentence_list = self.__encoder_tokenizer(encoder_sentence_list)
        decoder_sentence_list = self.__decoder_tokenizer(decoder_sentence_list)

        # setting encoder, decoder number
        self.encoder_max_number = max([len(sentence) for sentence in encoder_sentence_list])
        self.decoder_max_number = max([len(sentence) for sentence in decoder_sentence_list])
        self.max_encoder_vocab_number = len(self.encoder_data_dic)
        self.max_decoder_vocab_number = len(self.decoder_data_dic)

    def sentence_apply_padding_and_vector(self, sentence):
        encoder_sentence_list = self.__encoder_tokenizer([sentence])
        encoder_sentence_list = self.__make_encoder_padding(encoder_sentence_list)
        result_encoder = self.__apply_one_hot_vector(encoder_sentence_list, self.encoder_data_dic)
        decoder_sentence_list = [[self.GO] + [self.PAD for _ in range(self.decoder_max_number)]]
        result_decoder = self.__apply_one_hot_vector(decoder_sentence_list, self.decoder_data_dic)

        return result_encoder, result_decoder

    def __make_dic(self, encoder_sentence_list, decoder_sentence_list):
        encoder_set = set([self.UNK, self.PAD])
        for encoder_sentence in encoder_sentence_list:
            encoder_set.update(encoder_sentence)
        self.encoder_data_dic = {encoder: i for i, encoder in enumerate(encoder_set)}
        self.max_encoder_vocab_number = len(self.encoder_data_dic)

        decoder_set = set([self.PAD, self.GO, self.EOS])
        for decoder_sentence in decoder_sentence_list:
            decoder_set.update(decoder_sentence)
        self.decoder_data_dic = {decoder: i for i, decoder in enumerate(decoder_set)}
        self.max_decoder_vocab_number= len(self.decoder_data_dic)

    def get_encoder_dic(self):
        return self.encoder_data_dic

    def get_decoder_dic(self):
        return self.decoder_data_dic

    def __apply_one_hot_vector(self, total_padding_list, word_dic, target_mode=False):
        par_max_vector_length = len(word_dic)
        if target_mode is False:
            par_apply_one_hot_vector = lambda x, dic: np.eye(par_max_vector_length)[dic[x]]
        else:
            par_apply_one_hot_vector = lambda x, dic: dic[x]

        one_hot_vector_list = []
        for padding_list in total_padding_list:
            sentence_list = []
            for pad in padding_list:
                try:
                    pad = par_apply_one_hot_vector(pad, word_dic)
                except KeyError:
                    pad = par_apply_one_hot_vector(self.UNK, word_dic)
                sentence_list.append(pad)
            one_hot_vector_list.append(sentence_list)
        return one_hot_vector_list

    def __encoder_tokenizer(self, encoder_sentence_list):
        for i, sentence in enumerate(encoder_sentence_list):
            sentence = clean_str(sentence)
            encoder_sentence_list[i] = nltk.word_tokenize(sentence)
        return encoder_sentence_list

    def __decoder_tokenizer(self, decoder_sentence_list):
        for i, sentence in enumerate(decoder_sentence_list):
            decoder_sentence_list[i] = sentence.split('-')
        return decoder_sentence_list

    def __make_encoder_padding(self, encoder_sentence_list):
        for i, char_list in enumerate(encoder_sentence_list):
            remainder = self.encoder_max_number - len(char_list)
            char_list = char_list + [self.PAD for _ in range(remainder)]
            char_list.reverse()
            encoder_sentence_list[i] = char_list
        return encoder_sentence_list

    def __make_decoder_padding(self, decoder_sentence_list):
        decoder_sentence_list = copy.deepcopy(decoder_sentence_list)
        for i, char_list in enumerate(decoder_sentence_list):
            remainder = self.decoder_max_number - len(char_list)
            char_list = [self.GO] + char_list + [self.PAD for _ in range(remainder)]
            decoder_sentence_list[i] = char_list
        return decoder_sentence_list

    def __make_target_padding(self, target_sentence_list):
        target_sentence_list = copy.deepcopy(target_sentence_list)
        for i, char_list in enumerate(target_sentence_list):
            remainder = self.decoder_max_number - len(char_list)
            char_list = char_list + [self.EOS] + [self.PAD for _ in range(remainder)]
            target_sentence_list[i] = char_list
        return target_sentence_list

    def __split_data_set(self, data_set):
        for i, sentence in enumerate(data_set):
            if i % 2 == 0:
                self.total_data_dic.setdefault('q', []).append(sentence.strip())
            else:
                self.total_data_dic.setdefault('a', []).append(sentence.strip())

    def get_encoder_size(self):
        return self.encoder_max_number

    def get_decoder_size(self):
        # ['_GO_'] and ['_EOS_'] => (+1)
        return self.decoder_max_number + 1

    def get_encoder_vocab_size(self):
        return self.max_encoder_vocab_number

    def get_decoder_vocab_size(self):
        return self.max_decoder_vocab_number

    def iter_batch(self, epochs=10, batch_size=4):
        par_batch_repeat = int(self.total_sentence_number / batch_size) + 1
        for epoch in range(epochs):
            for par in range(par_batch_repeat):
                start = par * batch_size
                end = min((par+1) * batch_size, self.total_sentence_number)
                if start == end:
                    continue
                yield (
                    np.array(self.encoder_applied_padding_and_vector[start:end]),
                    np.array(self.decoder_applied_padding_and_vector[start:end]),
                    self.target_applied_padding_and_vector[start:end]
                )


class MakeDictionary(object):
    def __init__(self,
                 encoder_data_dir,
                 decoder_data_dir,
                 encoder_dic_dir,
                 decoder_dic_dir):
        """
        :param encoder_data_dir: encoder data directory
        :param decoder_data_dir: decoder data directory
        :param encoder_dic_dir:  save encoder dictionary directory
        :param decoder_dic_dir:  save decoder dictionary directory
        """

        self.encoder_file_list = [os.path.join(encoder_data_dir, file_name) for file_name in os.listdir(encoder_data_dir)]
        self.decoder_file_list = [os.path.join(decoder_data_dir, file_name) for file_name in os.listdir(decoder_data_dir)]
        self.encoder_dic_dir = encoder_dic_dir
        self.decoder_dic_dir = decoder_dic_dir

        self.PAD = '__PAD__'
        self.EOS = '__EOS__'
        self.GO = '__GO__'
        self.UNK = '__UNK__'
        self.padding_list = [self.PAD, self.EOS, self.GO, self.UNK]

    def get_padding_list(self):
        """
        :return: get padding_list : ['__PAD__', '__EOS__', '__GO__', '__UNK__'] 
        """
        return self.padding

    def make_dic(self):
        encoder_counter = collections.Counter()
        for encoder_filename in self.encoder_file_list:
            lines = [nltk.word_tokenize(clean_str(line)) for line in open(encoder_filename).readlines()]
            for word in list(itertools.chain(*lines)):
                encoder_counter[word.strip()] += 1
            del lines

        decoder_counter = collections.Counter()
        for decoder_filename in self.decoder_file_list:
            lines = [line.split('-') for line in open(decoder_filename).readlines()]
            for word in list(itertools.chain(*lines)):
                decoder_counter[word.strip()] += 1
            del lines
        encoder_filename = open(self.encoder_dic_dir, 'wb')
        decoder_filename = open(self.decoder_dic_dir, 'wb')
        pickle.dump(encoder_counter, encoder_filename)
        pickle.dump(decoder_counter, decoder_filename)
        encoder_filename.close()
        decoder_filename.close()

    def load_dic(self):
        encoder_filename = open(self.encoder_dic_dir, 'rb')
        decoder_filename = open(self.decoder_dic_dir, 'rb')
        encoder_dic = pickle.load(encoder_filename)
        decoder_dic = pickle.load(decoder_filename)
        encoder_filename.close()
        decoder_filename.close()

        encoder_sorted = sorted(encoder_dic.keys(), key=operator.itemgetter(0))
        decoder_sorted = sorted(decoder_dic.keys(), key=operator.itemgetter(0))
        encoder_sorted.extend(self.padding_list)
        decoder_sorted.extend(self.padding_list)

        encoder_dic = self.__make_dic(encoder_sorted)
        decoder_dic = self.__make_dic(decoder_sorted)

        return encoder_dic, decoder_dic

    @staticmethod
    def __make_dic(chars_lists):
        temp_dic = {}
        num = 0
        for char in chars_lists:
            temp_dic[char] = num
            num = num+1
        return temp_dic


class AdvPreProcessing(object):
    def __init__(self,
                 session,
                 enc_dic,
                 dec_dic,
                 enc_sequence_length,
                 dec_sequence_length,
                 contents_file_dir=None,
                 label_file_dir=None,
                 total_data_len=None,
                 batch_size=None):
        """
        :param session: setting tensor flow setting
        :param enc_dic: setting encoder dic (type dic)
        :param dec_dic: setting decoder dic (type dic)
        :param enc_sequence_length: encoder sequence length
        :param dec_sequence_length: decoder sequence length
        :param contents_file_dir: content file directory
        :param label_file_dir: label file directory
        :param total_data_len: total data set count
        :param batch_size: batch size ( defalut : 4 )
        """

        self.enc_dic = enc_dic
        self.dec_dic = dec_dic
        self.enc_sequence_length = enc_sequence_length
        self.dec_sequence_length = dec_sequence_length
        self.session = session
        self.total_data_set_len = total_data_len
        self.batch_size = batch_size

        if contents_file_dir is None:
            print('mode apply sentence')
            return

        contents_file_name_list = [os.path.join(contents_file_dir, file_name)
                                   for file_name in os.listdir(contents_file_dir)]
        labels_file_name_list = [os.path.join(label_file_dir, file_name)
                                 for file_name in os.listdir(label_file_dir)]

        contents_filename_queue = tf.train.string_input_producer(contents_file_name_list, shuffle=False)
        labels_filename_queue = tf.train.string_input_producer(labels_file_name_list, shuffle=False)

        contents_reader = tf.TextLineReader()
        labels_reader = tf.TextLineReader()

        _, contents_value = contents_reader.read(contents_filename_queue)
        _, labels_value = labels_reader.read(labels_filename_queue)

        self.contents, self.labels = tf.train.batch(
            [contents_value, labels_value],
            batch_size=self.batch_size,
            num_threads=FLAGS.batch_num_threads,
            name='batch_processing'
        )

    def __iter__(self):
        return self

    def __next__(self):
        encoder_input, decoder_input, decoder_target = self.__padding(self.contents, self.labels)
        return encoder_input, decoder_input, decoder_target

    def apply_sentence_to_padding(self, sentence):
        char_list = nltk.tokenize.word_tokenize(sentence)
        encoder_padding_list = []
        decoder_padding_list = []

        for char in char_list:
            try:
                encoder_padding_list.append(self.enc_dic[char])
            except KeyError:
                encoder_padding_list.append(self.enc_dic['__UNK__'])
        encoder_padding_list.append(self.enc_dic['__EOS__'])
        encoder_padding_list.extend([self.enc_dic['__PAD__']
                                     for _ in range(self.enc_sequence_length - len(encoder_padding_list))])

        decoder_padding_list.append(self.dec_dic['__GO__'])
        decoder_padding_list.extend([self.dec_dic['__PAD__'] for _ in range(self.dec_sequence_length - 1)])

        return encoder_padding_list, decoder_padding_list

    def __padding(self, content, label):
        content = content.eval(session=self.session)
        label = label.eval(session=self.session)

        for i in range(self.batch_size):
            content[i] = nltk.word_tokenize(clean_str(content[i].decode('utf-8')))
            label[i] = label[i].decode('utf-8').split('-')

        encoder_input = self.__apply_encoder_dic(content)
        decoder_input, decoder_target = self.__apply_decoder_dic(label)
        return encoder_input, decoder_input, decoder_target

    def __apply_encoder_dic(self, content):
        new_temp = []
        for i in range(self.batch_size):
            temp = []
            try:
                for char in content[i]:
                    temp.append(self.enc_dic[char])
            except KeyError:
                temp.append(self.enc_dic['__UNK__'])
            temp.append(self.enc_dic['__EOS__'])
            temp.reverse()
            t = [self.enc_dic['__PAD__'] for _ in range(self.enc_sequence_length - len(temp))]
            t.extend(temp)
            del temp
            new_temp.append(t)
        return new_temp

    def __apply_decoder_dic(self, labels):
        import copy
        target = []
        new_temp = []
        for i in range(self.batch_size):
            temp = []
            for char in labels[i]:
                temp.append(self.dec_dic[char])
            target_temp = copy.deepcopy(temp)

            temp.insert(0, self.dec_dic['__GO__'])
            temp.extend([self.dec_dic['__PAD__'] for _ in range(self.dec_sequence_length - len(temp))])
            new_temp.append(temp)

            target_temp.append(self.dec_dic['__EOS__'])
            target_temp.extend([self.dec_dic['__PAD__'] for _ in range(self.dec_sequence_length - len(target_temp))])
            target.append(target_temp)
        return new_temp, target


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()