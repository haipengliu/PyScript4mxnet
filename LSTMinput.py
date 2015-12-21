# pylint:skip-file
import numpy as np
from numpy import array

class LSTMinput(object):

    def __init__(self, input_file='', sent_len=35):
        self.infn = input_file
        self.sent_len = sent_len
        print ("===== LSTM input factory module =====")
        print ("===== by haipeng liu. =====")
        print ("Created a LSTM input object from input corpus file %s." % input_file)
        return

    def __del__(self):
        return

    def make_lstm_input(self, long_indexes, batch_size=20, sentence_len=35):
        """ Given a long indexes list, reshape it to a numpy matrix as input to LSTM.
            Matrix shape is ((nbatch * sentence_len) * batch_size)
            Note: the reshaped matrix will be in numpy.
                  The length of axis 1 is the batch_size.
                  The length of axis 0 is the nbatch * sentence_len
                  
        Params
        ------
        long_indexes :  list object
            the list of all sentences concatenating together

        batch_size : int
            how many sentences in one batch

        sentence_len : int
            how many words in a sentence
        
        Return
        ------
        A numpy matrix shaped as nbatch * batch_size.        

        """
        nTotalWords = len(long_indexes)
        nSentence = (nTotalWords / sentence_len)
        nBatch = (nSentence / batch_size)
        nIgnore = (nSentence % batch_size)
        print("==========================================================")
        print("Making the input to LSTM.")
        print("There are %d words in the given long list." % len(long_indexes))
        print("There are %d words in one sentence." % sentence_len)
        print("There are %d sentences in the given long list." % nSentence)
        print("There are %d sentences in one batch." % batch_size)
        print("There will be %d batches." % nBatch)
        print("There are %d sentences that will be kept." % (nBatch * batch_size))
        print("There are %d sentences will be throw away." % nIgnore)
        long_indexes_cut = long_indexes[ : (-nIgnore * sentence_len)]
        print("After throwing away. There are %d sentences left."
              % (len(long_indexes_cut) / sentence_len))
        print("After throwing away. There are %d total words in long list."
              % len(long_indexes_cut))
        print("Copying the cut long list into numpy .........")
        nd_long_indexes = array(long_indexes_cut)
        nbatch_mul_senten_len = int(nd_long_indexes.shape[0] / batch_size)
        print("Reshaping the cut long list to ( %d * %d )." % (nbatch_mul_senten_len, batch_size))
        nd_mat = nd_long_indexes.reshape((nbatch_mul_senten_len, batch_size), order='F')
        return nd_mat

    def filterCorpus2LongIndexes(self, dict):
        """ filter the corpus from words to indexes with given dict.
            Then append each index line into a super long line

        Params
        ------
        dict : dict
            given dict to filter the corpus

        Return
        ------
        A super long line concatenation all the lines from corpus
        
        """
        fi = open(self.infn)
        lines = fi.readlines()
        fi.close()
        print("==========================================================")
        print("Filter the corpus %s with dict. Convert words into indexes."
              % self.infn)
        print("The input file has %d lines." % len(lines))
        long_indexes = []
        nb_filtered_lines = 0
        line_num = 0
        for line in lines:
            line_num += 1
            line = line.strip()
            words = line.split(' ')
            if len(words) < self.sent_len:
                error_words = words[:]
                for i in xrange(self.sent_len - len(words)): 
                    words.append('<EOS>')
            try:
                assert len(words) == self.sent_len
            except AssertionError, e:
                print("The error line is %s" % str(error_words))
                print("The error line has %d words." % len(words))
                print("The line number is %d." % line_num)
                print("The padded line is %s." % str(words))
            indexes = []
            for word in words:
                if dict.get(word) == None:
                    indexes.append(dict['<UNK>'])
                else:
                    indexes.append(dict[word])
            long_indexes += indexes
            nb_filtered_lines += 1
        print("Has filtered %d lines." % nb_filtered_lines)
        print("The filtered lines are concatenated to a super long line with %d words."
              % len(long_indexes))
        return long_indexes

    def save_LSTM_input(self, nd_mat, sfn):
        m, n = nd_mat.shape
        sfi = open(sfn, 'w')
        print("==========================================================")
        print("Saving the LSTM input matrix into file %s." % sfn)
        print("The matrix has %d rows and %d cols." % (m, n))
        for i in xrange(m):
            for j in xrange(n):
                sfi.write(str(nd_mat[i][j]) + '  ')
            sfi.write('\n')
        sfi.close()
        print("Saving done.")
        return
            
    def load_LSTM_input(self, lfn):
        print("==========================================================")
        print("Loading the LSTM input matrix from file %s." % lfn)
        lfi = open(lfn)
        ct = lfi.readlines()
        lfi.close()
        m = len(ct)
        n = len(ct[0])
        print("The file %s has %d rows and %d cols." % (lfn, m, n))
        print("So allocating a %d * %d ndarray." % (m, n))
        ndmat = np.zeros( ( m, n ) )
        x = 0
        y = 0
        print("Filling the ndarray from %dth rows and %dth cols." % (x, y))
        print(".............")
        for line in ct:
            line = line.strip()
            indexes = line.split(' ')
            y = 0
            for index in indexes:
                ndmat[x][y] = indexes
                y += 1
            x += 1
        p, q = ndmat.shape
        print("Filled to %dth row and %dth col." % (x, y))
        print("After filling, the matrix has %d rows and %d cols." % (p, q))
        return ndmat
