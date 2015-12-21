# pylint:skip-file
import sys
sys.path.insert(0, "../../python")
sys.path.append("..")
import lstm
import mxnet as mx
import numpy as np
from corpusProcess import corpus
from LSTMinput import LSTMinput

# data parms
data_path = '../kaldi_data'
input_file = data_path + '/librispeech-lm-norm.train.txt'
dict_file = data_path + '/librispeech.train.dict'
word_count_file = data_path + '/librispeech.train.wc'
small_file_prefix = data_path + '/librispeech.train.small'
max_vocab = 20000 # size limit [1, 100000)
batch_size = 40
sentence_len = 35
nb_split = 100
mat_indexes_file_prefix = data_path + '/librispeech.train.inputMat.small'

# valid
valid_file = data_path + '/valid/librispeech-lm-norm.valid.txt'
valid_format_file = data_path + '/valid/librispeech-lm-norm.format.txt'
valid_indexes_file = data_path + '/valid/librispeech-lm-norm.valid.inputMat.txt'

# LSTM params
num_hidden = 200
num_embed = 200
num_lstm_layer = 2
num_round = 25
learning_rate= 0.1
wd=0.
momentum=0.0
max_grad_norm = 5.0
update_period = 1



cp = corpus(input_file)
big_lines = cp.format_lines(sentence_len)
dict = cp.make_dict(big_lines, max_vocab, saveWordCount=True, wcfn=word_count_file)
cp.save_dict(dict, dict_file)
cp.seperateBigFile(big_lines, nb_split, saveSmall=True, prefix=small_file_prefix)

#dict = cp.load_dict(dict_file)


###### train ######
for i in xrange(nb_split):
    input_small_file = small_file_prefix + '.' + str(i)
    mat_indexes_file = mat_indexes_file_prefix + '.' + str(i)
    LIN = LSTMinput(input_small_file, sentence_len)
    long_indexes = LIN.filterCorpus2LongIndexes(dict)
    mat_indexes = LIN.make_lstm_input(long_indexes, batch_size, sentence_len)
    LIN.save_LSTM_input(mat_indexes, mat_indexes_file)
    del LIN

###### valid ######
cp_valid = corpus(valid_file)
big_valid_lines = cp_valid.format_lines(sentence_len)
cp_valid.seperateBigFile(big_valid_lines, 1, saveSmall=True, prefix=valid_format_file)

LIN_valid = LSTMinput(valid_format_file+'.0', sentence_len)
long_indexes_valid = LIN_valid.filterCorpus2LongIndexes(dict)
mat_indexes_valid = LIN_valid.make_lstm_input(long_indexes_valid, batch_size, sentence_len)
LIN_valid.save_LSTM_input(mat_indexes_valid, valid_indexes_file)
X_val = LIN_valid.load_LSTM_input(valid_indexes_file)


model = lstm.setup_rnn_model(mx.gpu(),
                             num_lstm_layer=num_lstm_layer,
                             seq_len=sentence_len,
                             num_hidden=num_hidden,
                             num_embed=num_embed,
                             num_label=max_vocab,
                             batch_size=batch_size,
                             input_size=max_vocab,
                             initializer=mx.initializer.Uniform(0.1),dropout=0.5)


for i in xrange(nb_split):
    input_small_file = small_file_prefix + '.' + str(i)
    mat_indexes_file = mat_indexes_file_prefix + '.' + str(i)
    LIN = LSTMinput(input_small_file, sentence_len)
    X_train = LIN.load_LSTM_input(mat_indexes_file)
    lstm.train_lstm(model, X_train, X_val,
                    num_round=num_round,
                    half_life=2,
                    max_grad_norm = max_grad_norm,
                    update_period=update_period,
                    learning_rate=learning_rate,
                    wd=wd)
    del X_train
    del LIN

    
