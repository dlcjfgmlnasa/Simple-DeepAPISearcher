# data_helper.py
import os
import nltk
import re
import copy
import numpy as np


class PreDataProcessing(object):
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

        self.encoder_applied_padding_and_vector = []
        self.decoder_applied_padding_and_vector = []
        self.target_applied_padding_and_vector = []

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

    def sentence_apply_padding_and_vector(self, sentence):
        encoder_sentence_list = self.__encoder_tokenizer(sentence)
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
            sentence = PreDataProcessing.__clean_str(sentence)
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

    def make_dic(self, file_dir):
        pass

    def load_dic(self):
        pass

    @staticmethod
    def __clean_str(string):
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