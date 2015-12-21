# pylint:skip-file
from collections import defaultdict
from collections import Counter


class corpus(object):
    def __init__(self, input_file=''):
        self.infn = input_file
        print ("===== Corpus Process module =====")
        print ("===== by haipeng liu. =====")
        print ("Created a corpus object from input file %s." % input_file)
        return

    def __del__(self):
        return

    def make_dict(self, lines, max_vocab=200000, saveWordCount=False, wcfn=''):
        ''' Make dict from formated text(lines).

        Params
        ------
        lines : list
            list of lines formated from corpus.format_lines() function

        max_vocab : int
            the maximum number of keys in dict

        saveWordCount : boolean
            whether to save the dict into a file

        wcfn : str
            the word count file name used to save the dict

        Return
        ------
        dict : dict
            the dict object
        '''
        all_words = []
        for line in lines:
            line = line.strip()
            words = line.split(' ')
            all_words += words
        cnt = Counter(all_words)
        common = cnt.most_common(max_vocab)
        print ("=============================================")
        print("There are total %d words in corpus." % len(common))
        print ("The first %d words will be used to make dict." % max_vocab)
        dict = defaultdict(int)
        dict['<UNK>'] = 0
        dict['<EOS>'] = 1
        idx = 2

        if saveWordCount == True:
            wcfi = open(wcfn, 'w')
            print("Will save word count file in %s" % wcfn)
            
        for w, c in common:
            if saveWordCount == True and wcfi:
                wcfi.write(w + '\t' + str(c) + '\n')
            dict[w] = idx
            idx += 1
        if saveWordCount == True and wcfi:
            wcfi.close()
        return dict

    def save_dict(self, dict, fn=''):
        fi = open(fn, 'w')
        for key in dict.keys():
            fi.write(key + '\t' + str(dict[key]) + '\n')
        print("The corpus dict has been saved in %s" % fn)
        fi.close()

    def load_dict(self, fn=''):
        print("==========================================================")
        print("Loading dict from %s." % fn)
        fi = open(fn)
        dict = {}
        ct = fi.readlines()
        print("There are %d KV pairs in dict %s." % (len(ct), fn))
        for line in ct:
            line = line.strip()
            pair = line.split('\t')
            dict[pair[0]] = pair[1]
        fi.close()
        print("There are %d KV paris in loaded dict." % len(dict))
        return dict

    def format_lines(self, nb_words=40):
        '''Cut the lines if they are long. Pad the lines if they are short

        Params
        ------
        nb_words : int
            how many words in a line after format

        Return
        ------
        newlines : list
            list of new lines
        '''
        fi = open(self.infn)
        lines = fi.readlines()
        fi.close()
        newlines = []
        print ("=============================================")
        print("Format the corpus line with the word limit number %d." % nb_words)
        print("Cut if too long. Pad with '<EOS>' if too short.")
        print("There are %d lines before format." % len(lines))
        for line in lines:
            line = line.strip()
            words = line.split(' ')
            words_cut = words[ 0 : nb_words ]
            if len(words_cut) < nb_words:
                pad_len = nb_words - len(words_cut)
                for i in xrange(pad_len):
                    words_cut.append('<EOS>')
            newline = ' '.join(words_cut)
            newlines.append(newline)
        print("There are %d new lines after format." % len(newlines))
        return newlines
        
    def seperateBigFile(self, big_lines, nb_split=10, saveSmall=False, prefix=''):
        ''' This function will seperate a big file into small files.

        Parameters
        ----------
        
        big_lines : list
            the lines from big file by readlines() function

        nb_split : int
            the number of splits you want to split
        
        saveSmall : boolean
            whether to save the splited small files

        prefix : str
            if saving small files, give the prefix as the file name

        Returns
        -------
        list_small_lines : list
            list of lines of small files
        '''

        print ("=============================================")
        print("Begin seperating the big file into small ones.")
        big_len = len(big_lines)
        small_len = big_len / nb_split
        print("There are %d lines in the big file." % big_len)
        print("Split the big file into %d small ones." % nb_split)
        print("Each small file will have %d lines." % small_len)
        if saveSmall == True and prefix:
            print("Will save small files with file name prefixed as %s." % prefix)
        for step in xrange(nb_split):
            start = step * small_len
            end = start + small_len
            small_lines = big_lines[ start : end ] # from start to end - 1
            if saveSmall == True and prefix:
                small_fn = prefix + '.' + str(step)
                print("Save the lines from %d to %d into the small file %s." \
                      % (start, end - 1, small_fn))
                small_fi = open(small_fn, 'w')
                for line in small_lines:
                    small_fi.write(line + '\n')
                small_fi.close()
        return        
        
