# -*- coding: utf-8 -*-
'''
greedy segmenter based on embedding matching.

To reproduce the ACL paper results, run script2.py with the config.txt file:
python script2.py config.txt

The training takes about 1 hour using a single core of Intel Core5 @1.9GHz.
'''

from word2vec2 import Word2Vec, Vocab
#from gensim import matutils, utils
#from scipy import special
from random import shuffle, randint as r_randint
from numpy import  exp, dot, zeros, outer, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, append as np_append, asanyarray, arange, multiply, copy as np_copy, isnan as np_isnan
from numpy.random import permutation
import codecs, sys, time, logging, os, math, datetime

from six import iteritems, itervalues, string_types
logger = logging.getLogger("gensim.models.word2vec")

#from batch_seg_eval_sum import parse_evaluation_result

def parse_evaluation_result(path_to_evaluation_result):
    f=codecs.open(path_to_evaluation_result, 'rU', 'utf-8')
    lines=f.readlines()
    d_str=lines[-1]
    last_line_tokens=d_str.split()
    #last_line_tokens=f.readlines()[-1].split()
    if last_line_tokens[0]=='###' and len(last_line_tokens)==14:
        recall, precision,f_score,oov_rate,oov_recall,iv_recall=[float(i) if i!="--" else i for i in last_line_tokens[-6:]]
        return (path_to_evaluation_result.split('/')[-1], f_score, oov_recall,iv_recall, recall, precision)
    else:
        print('error! Format of the EVALUATION RESULT does not match the standard!')




def full2halfwidth(uchar):
    if uchar == u'':
        return u''
    inside_code = ord(uchar)

    if not inside_code in range(65280, 65375):
        return uchar

    else:
        inside_code -= 65248
        return unichr(inside_code)


def emb_normalization(model):
    #if Flag:
    #    table = model.syn0
    #else:
    #    table = model.syn1neg
    print 'normalization of pre-trained embedding...'
    for table in [model.syn0, model.syn1neg]:

        norm_list=[]

        for i in xrange(table.shape[0]):
            norm = sqrt((table[i, :] ** 2).sum(-1))
            norm_list.append(norm)

            #print '\nword=', model.index2word[i], 'norm=',norm
            if norm>6.0:
                table[i, :] /= (norm/6.0)

            elif norm<1.0:
                table[i, :] /= norm
                #table[i,:] /= 2.0
                #print '\tnew norm=', sqrt((table[i, :] ** 2).sum(-1))
        norm_list.sort()
        print '\n\n==>major statistics of norm: min=', norm_list[0], 'max=', norm_list[-1],\
            'median=', norm_list[len(norm_list)/2], 'avg=', sum(norm_list)/len(norm_list)
        print 'first 1/4=', norm_list[len(norm_list)/4], 'last 1/4=', norm_list[-len(norm_list)/4]

        #f= codecs.open('norm.list.'+str(time.time())[-6:]+'.txt','w','utf-8')
        #n2=map(str, norm_list)
        #f.write(', '.join(n2)+'\n')
        #f.close()

class Seger(Word2Vec):
    def __init__(self, size=50, alpha=0.1, min_count=1, seed=1, workers=1,iter=1, use_gold=0, train_path =None,
                 test_raw_path = None, test_path = None, dev_path = None, quick_test = None, dict_path = None,
                 score_script_path = None, pre_train = False, uni_path = None, bi_path = None, hybrid_pred=False,
                 no_action_feature=False, no_bigram_feature = False, no_unigram_feature=False, no_binary_action_feature=False, no_sb_state_feature=False, **kwargs):

        print '\n\n### Initialization of the segmentation model ###'

        self.no_action_feature = no_action_feature
        self.no_bigram_feature = no_bigram_feature
        self.no_unigram_feature= no_unigram_feature
        self.no_binary_action_feature= no_binary_action_feature
        self.no_sb_feature = no_sb_state_feature


        self.pre_train = pre_train
        self.l2_rate = 0.001 # rate for L2 regularization

        if self.l2_rate:
            print 'reg with L2, with param=', self.l2_rate
        self.drop_out = False



        self.finger_int=str(r_randint(0,1000000))
        self.binary_pred = False
        self.hybrid_pred = hybrid_pred

        self.use_gold = use_gold
        #self.model = None
        self.START = "#S#"
        self.END ="#E#"

        self.label0_as_vocab, self.label1_as_vocab, self.unknown_as_vocab="$LABEL0", "$LABEL1", "$OOV"

        self.su_prefix, self.sb_prefix ='$SU','$SB' #prefix for unigram/bigram state; no prefix for *char* unigram/bigrams
        self.state_varient=('0','1')


        self.train_path = train_path
        self.test_raw_path= test_raw_path
        self.test_path = test_path
        self.dev_path = dev_path
        self.quick_test= quick_test
        self.dict_path = dict_path
        self.score_script = score_script_path

        #self.score_script = '../working_data/score'
        #self.dict_path ='../working_data/pku.dict'

        print '\nloading train, test, dev corpus...'

        self.train_corpus = [l.split() for l in codecs.open(self.train_path, 'rU','utf-8')]
        self.test_corpus=[l.split() for l in codecs.open(self.test_raw_path, 'rU', 'utf-8')]
        self.dev_corpus =[l.split() for l in codecs.open(self.dev_path,'rU', 'utf-8')]
        self.quick_test_corpus =[l.split() for l in codecs.open(self.quick_test,'rU', 'utf-8')]

        Word2Vec.__init__(self, sentences= None, size=size, alpha=alpha, min_count=min_count,seed=seed, workers=workers, iter=iter, **kwargs)

        self.mask=[1 for i in range(12)]

        if self.no_action_feature:
            self.mask=self.mask[:-3]
            print 'len mask', len(self.mask)
        elif self.no_sb_feature:
            self.mask=self.mask[:-1]
            print 'len mask', len(self.mask)

        if self.no_unigram_feature:
            self.mask= self.mask[:-5]
            print 'len mask', len(self.mask)

        if self.no_bigram_feature:
            self.mask = self.mask[:-4]
            print 'len mask', len(self.mask)


        self.f_factor = sum(self.mask)
        self.f_factor2= 2

        if self.no_binary_action_feature:
            self.f_factor2 = 0
            print 'f-factor2=', self.f_factor2

        self.non_fixed_param = self.f_factor*self.layer1_size
        self.pred_size = self.non_fixed_param+self.f_factor2

        if self.drop_out:
            self.dropout_rate=0.5
            self.dropout_size= int(self.dropout_rate*self.non_fixed_param)
            print 'using drop_out, rate/size=', self.dropout_rate, self.dropout_size

        self.train_mode = False

        self.dev_test_result=[]

        print '\nLearning rate=', self.alpha,'; Feature (layer1) size=', self.layer1_size, '; Predicate vec size=', self.pred_size, 'f-factor=',self.f_factor, 'f-factor2=', self.f_factor2


        if self.pre_train:

            print '\nloading pre-trained char and char-bigram embeddings'
            self.uni_emb=Word2Vec.load(uni_path)
            emb_normalization(self.uni_emb)
            print 'unigram embedding loaded'
            self.bi_emb = Word2Vec.load(bi_path)
            emb_normalization(self.bi_emb)
            print 'bigram embedding loaded'




    def predict_sigle_position(self, sent, pos, prev2_label, prev_label):

        flag = False

        feature_vec, feature_index_list = self.gen_feature(sent, pos, prev2_label, prev_label)

        if self.train_mode and self.drop_out:

            to_block=set(permutation(arange(self.non_fixed_param))[:self.dropout_size])
            #print 'to_block',list(to_block)[:10]
            block = array([0 if zzz in to_block else 1 for zzz in range(self.pred_size)])

            feature_vec = multiply(feature_vec, block)

        elif self.drop_out: # for dropout mode at testing time...
            feature_vec = (1-self.dropout_rate) * feature_vec
            block = None

        else:
            block = None


        if block:
            print 'block=', block



        if flag:
            print 'pos, char=', pos, sent[pos]
            print 'feat_index_list=', feature_index_list, ';features are:', ' '.join([self.index2word[ind] for ind in feature_index_list])

        c0 = sent[pos] if pos<len(sent) else self.END

        pred_tuple = tuple([self.su_prefix+varient+c0 for varient in self.state_varient])
        if pred_tuple[0] in self.vocab and pred_tuple[1] in self.vocab:
            pass
        else:
            pred_tuple = None
            if self.train_mode:
                print 'Unknown candidate! Should NOT happen during training!'
                assert False

        pred_tuple2 = tuple([self.label0_as_vocab, self.label1_as_vocab])

        softmax_score = None
        if pred_tuple:
            pred_index_list = [self.vocab[pred].index for pred in pred_tuple]
            pred_matrix = self.syn1neg[pred_index_list]

            if block is not None:
                pred_matrix = multiply(block, pred_matrix)

            elif self.drop_out:
                pred_matrix = (1-self.dropout_rate) * pred_matrix



            raw_score = exp(dot(feature_vec, pred_matrix.T))
            softmax_score= raw_score/sum(raw_score)


        pred_index_list2 = [self.vocab[pred].index for pred in pred_tuple2]
        pred_matrix2 = self.syn1neg[pred_index_list2]

        if block is not None:
            pred_matrix2 = multiply(block, pred_matrix2)
        elif self.drop_out:
            pred_matrix = (1-self.dropout_rate) * pred_matrix2

        raw_score2 = exp(dot(feature_vec, pred_matrix2.T))
        softmax_score2= raw_score2/sum(raw_score2)
        #print pred_matrix2.shape, pred_matrix.shape
        if pred_tuple:
            softmax_score2 = np_append(softmax_score2, softmax_score)
            pred_index_list2.extend(pred_index_list)
            pred_matrix2 = np_append(pred_matrix2, pred_matrix, axis=0)
            #print pred_matrix2.shape, pred_matrix.shape

        if flag: print 'pred index and item=', pred_index_list2, ' '.join([self.index2word[ind] for ind in pred_index_list2])

        return softmax_score2, feature_index_list, pred_index_list2, feature_vec, pred_matrix2





    def train_gold_per_sentence(self, sentence, alpha, work=None):

        flag = False

        count_sum, error_sum = 0,  0.0

        if sentence:
            index, acc=[], 0
            for word in sentence:
                index.append(acc)
                acc += len(word)

            sentence="".join(sentence)

            count_sum = len(sentence)

            sentence=map(full2halfwidth, sentence)

            label_list = [1 for _ in range(len(sentence))]

            for i in index:
                label_list[i] = 0  # start-position-of-a-word is labeled as "1"

            prev2_label, prev_label = 0,0


            for pos in range(count_sum):

                #print '\npos', pos, 'char:',sentence[pos]

                softmax_score, feature_index_list, pred_index_list, l1, l2 \
                    = self.predict_sigle_position(sentence, pos, prev2_label, prev_label)

                #print 'l1 shape', l1.shape
                #print 'l2 shape', l2.shape

                if flag: print 'softmax_score=', softmax_score

                true_label = label_list[pos]

                if len(softmax_score)==4:
                    assert self.train_mode

                    if true_label == 0:
                        gold_score=[1.0, 0.0, 1.0, 0.0]

                    elif true_label == 1:
                        gold_score =[0.0, 1.0, 0.0, 1.0]
                    else:
                        print 'Error! true label should either 1 or 0, but now it is:', true_label

                else:
                    print 'The output of predict_single_position should have either 2 or 4 scores, but now it has '
                    assert False

                error_array=gold_score - softmax_score
                #print '\n\nerror, len_error_array', error_array, len(error_array)
                error_sum += sum(abs(error_array))/len(error_array)



                gb = error_array * alpha
                neu1e = zeros(self.non_fixed_param)


                if flag:
                    print 'fb, gb shape', softmax_score.shape, gb.shape
                    print 'feature/pred, indices size', len(feature_index_list), len(pred_index_list)
                    print 'outer(gb,l1) shape=', outer(gb, l1).shape



                #print '??',dot(gb, l2[:,0:self.non_fixed_param]).shape
                #print neu1e.shape
                #print np_sum(l2[:,0:self.non_fixed_param], axis=0).shape
                #print 'neu1e shape=', neu1e.shape, 'reshape:', neu1e.reshape(len(feature_index_list), len(neu1e)/len(feature_index_list)).shape,  '\n'
                #print 'l1 shape', l1.shape
                neu1e += dot(gb, l2[:,0:self.non_fixed_param])

                if self.l2_rate:
                    self.syn1neg[pred_index_list] = self.syn1neg[pred_index_list] - alpha*self.l2_rate*self.syn1neg[pred_index_list]
                    self.syn0[feature_index_list] = self.syn0[feature_index_list] -  alpha*self.l2_rate*self.syn0[feature_index_list]

                self.syn1neg[pred_index_list] += outer(gb, l1)
                self.syn0[feature_index_list] += neu1e.reshape(len(feature_index_list), len(neu1e)/len(feature_index_list))


                softmax_score = softmax_score[-2:]
                if softmax_score[1]>0.5:
                    label = 1
                else:
                    label = 0

                prev2_label = prev_label
                prev_label = label

                if self.use_gold:
                    prev_label = true_label

            #if random.random()>0.99:
            #    print 'error rate, n_count,n_error', error_sum/float(count_sum), count_sum ,error_sum





        return count_sum, error_sum






    def predict_sentence_greedy(self, sent):

        tokens = []
        if sent:
            old_sentence = "".join(sent)
            sentence = map(full2halfwidth, old_sentence) # all half-width version, used to predict label...

            prev2_label, prev_label = 0, 0


            for pos, char in enumerate(old_sentence): # char is still the char from original sentence, for correct eval
                if pos== 0:
                    label = 0
                else:
                    score_list, _, _, _, _ = self.predict_sigle_position(sentence, pos, prev2_label, prev_label)

                    if self.binary_pred:
                        score_list = score_list[:2]

                    elif self.hybrid_pred:
                        old_char = old_sentence[pos]
                        if old_char in self.vocab and self.vocab[old_char].count>self.hybrid_threshold:
                            score_list = score_list[-2:]
                        else:
                            #score_list = score_list[:2]
                            x,y= score_list[:2], score_list[-2:]
                            score_list = [(x[i]+y[i])/2.0 for i in range(2)]

                    else:
                        score_list = score_list[-2:]

                    #transform score to binary label
                    if score_list[1]>0.5:
                        label = 1
                    else:
                        label = 0

                    #print '\nscore, label=', score_list, label

                if label == 0:
                    tokens.append(char)

                else:
                    if tokens:
                        tokens[-1] += char
                    else:
                        tokens.append(char)
                        print 'should not happen! the action of the first char in the sent is "append!" '

                prev2_label = prev_label
                prev_label = label

        return tokens



    def segment_corpus(self, corpus):
        print 'segmenting corpus..'
        seg_corpus=[]
        for sent_no, sent in enumerate(corpus):
            if not sent_no%100:
                print 'num of sentence segmented:', sent_no, '...'
            seg_corpus.append(self.predict_sentence_greedy(sent))

        return seg_corpus

        print 'segmentation done'



    def eval(self, corpus, gold_path):
        seged_corpus = self.segment_corpus(corpus)
        time_str = self.finger_int+datetime.datetime.now().strftime('.%d.%H.%M')
        time_str='result.con.cat'+time_str
        tmp_path='../working_data/'+time_str+'.tmp.seg'
        tmp_eval_path=tmp_path+".eval"

        f= codecs.open(tmp_path,'w','utf-8')
        for sent in seged_corpus:
            f.write(' '.join(sent)+'\n')
        f.close()

        print 'eval with "score" '
        os.system("perl "+self.score_script+"  "+self.dict_path+" "+gold_path+"  "+tmp_path+"  >"+tmp_eval_path)
        _, f_score, oov_recall,iv_recall, recall, precision  = parse_evaluation_result(tmp_eval_path)

        if self.epoch<self.iter -1 or corpus == self.quick_test_corpus:
            os.system("rm "+tmp_eval_path)
            os.system("rm "+tmp_path)

        return f_score, oov_recall, iv_recall



    def gen_uni_gram_bigram(self,sent, pos):

        c0 = sent[pos] if pos<len(sent) else self.END
        c1 = sent[pos-1] if pos>0 else self.START
        c2 = sent[pos-2] if pos>1 else self.START
        c3 = sent[pos+1] if pos<len(sent)-1 else self.END
        c4 = sent[pos+2] if pos<len(sent)-2 else self.END

        b1, b2, b3, b4 = (c1,c0), (c2,c1), (c0,c3), (c3,c4)

        return c0, c1, c2, c3, c4, b1, b2, b3, b4


    def word2index(self, voc):
        if voc in self.vocab:
            index= self.vocab[voc].index
        else:
            index=self.vocab[self.unknown_as_vocab].index
            if self.train_mode:
                print 'Unknown vocabulary item:', voc, 'This should NOT happen during the training phase'
        return index



    def gen_feature(self,sent, pos, prev2_label, prev_label):

        c0, c1, c2, c3, c4, b1, b2, b3, b4 =  self.gen_uni_gram_bigram(sent, pos)

        ngram_feature_list= [c0, c1, c2, c3, c4]+[''.join(b) for b in [b1,b2,b3,b4]]

        if self.no_bigram_feature:
            ngram_feature_list =ngram_feature_list[:-4]

        if self.no_unigram_feature:
            ngram_feature_list = ngram_feature_list[5:]


        state_feature_list=[self.su_prefix+self.state_varient[int(item[0])]+item[1] for item in zip([prev2_label, prev_label], [c2, c1])]

        if not self.no_sb_feature:
            state_feature_list.append(self.sb_prefix+self.state_varient[int(prev_label)]+''.join(b1)) #change the bigram state def

        if self.no_action_feature:
            feat_list = ngram_feature_list
        else:
            feat_list = ngram_feature_list+state_feature_list

        feature_index_list = map(self.word2index, feat_list)
        feature_vec = self.syn0[feature_index_list].ravel()

        if self.no_binary_action_feature:
            feat_vec = feature_vec
        else:
            feat_vec=np_append(feature_vec, asanyarray([float(prev2_label), float(prev_label)]))  ###########  !!!!! tmp block the previous state feature..

        #print 'feature shape', feat_vec.shape
        #print 'feature_shape', feature_vec.shape
        #feat_vec=np_append(feature_vec, asanyarray([float(prev2_label), float(prev_label)]))  ###########  !!!!! tmp block the previous state feature..
        #print 'feature shape', feat_vec.shape

        return feat_vec, feature_index_list


    def do_training(self):

        if self.train_corpus:
            self.build_vocab(self.train_corpus)


            for n in range(self.iter):
                tic = time.time()
                self.train_mode = True
                self.epoch = n

                shuffle(self.train_corpus) #shuffle the corpus

                if self.total_words:
                    self.train(self.train_corpus, chunksize=200, total_words=self.total_words)
                else:
                    self.train(self.train_corpus, chunksize=200)

                self.train_mode = False

                print '\n===Eval on dev corpus at epoch', self.epoch, ':'
                f_score, oov_recall, iv_recall = self.eval(self.dev_corpus, self.dev_path)
                print 'F-score/OOV-Recall/IV-recall=', f_score, oov_recall, iv_recall

                dev_result = (f_score, oov_recall, iv_recall)



                print '\n===Eval on testing corpus:', self.epoch, ':'
                f_score, oov_recall, iv_recall = self.eval(self.test_corpus, self.test_path)
                print 'F-score/OOV-Recall/IV-recall=', f_score, oov_recall, iv_recall
                print '\n\n>>> Jobs done! Total time consumed:', time.time() - tic

                self.dev_test_result.append((dev_result, (f_score, oov_recall, iv_recall)))

            print '\n\n===Summary of Result aftr each epoch'
            for i, result in enumerate(self.dev_test_result):
                dev_result, test_result = result
                print '\nepoch ', i
                print 'dev f/o/i=', ', '.join(map(str, dev_result))
                print 'test fo/i=', ', '.join(map(str, test_result))

            if False:
                print '\n(Bonus)Evaluation on TEST Corpus with binary predicate:'
                self.binary_pred = True
                f_score, oov_recall, iv_recall = self.eval(self.test_corpus, self.test_path)
                print 'F-score/OOV-Recall/IV-recall=', f_score, oov_recall, iv_recall
            print '\n\n>>> Jobs done! Total time consumed:', time.time() - tic
            print '\n\n\n'







    def _vocab_from_new(self, sentences):
        sentence_no, vocab, vocab_pred = -1, {}, {}
        total_words = 0

        #for meta_subgram in [self.START, self.END]:
        #    vocab[meta_subgram]=Vocab(count =1)


        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % 200 == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words and %i word types" %
                            (sentence_no, total_words, len(vocab)))

            if sentence:

                char_seq=[self.START, self.START]+map(full2halfwidth, u"".join(sentence))+[self.END, self.END]

                total_words = total_words + len(char_seq)-3

                subgrams=[char for char in char_seq] +[self.su_prefix+varient+char for char in char_seq for varient in self.state_varient]
                bigrams =[char_seq[index]+char_seq[index+1] for index in range(len(char_seq)-1)]
                subgrams.extend([self.sb_prefix+varient+bigram for bigram in bigrams for varient in self.state_varient])
                subgrams.extend(bigrams)

                for sub in subgrams:
                    if sub in vocab:
                        vocab[sub].count += 1
                    else:
                        vocab[sub] = Vocab(count=1)


        logger.info("collected %i word types from a corpus of %i words and %i sentences" %
                    (len(vocab), total_words, sentence_no + 1))

        self.total_words = total_words

        return vocab

    def build_vocab(self, sentences):

        logger.info("collecting all words and their counts")
        vocab = self._vocab_from_new(sentences)

        # assign a unique index to each word
        self.vocab, self.index2word = {}, []


        for meta_word in [self.label0_as_vocab, self.label1_as_vocab, self.unknown_as_vocab]:

            v = Vocab(count=1)
            v.index = len(self.vocab)
            v.sample_probability = 1.0
            self.index2word.append(meta_word)
            self.vocab[meta_word] = v

        for subgram, v in iteritems(vocab):
            if v.count >= self.min_count:
                v.sample_probability = 1.0
                v.index = len(self.vocab)
                self.index2word.append(subgram)
                self.vocab[subgram] = v
        logger.info("total %i word types after removing those with count<%s" % (len(self.vocab), self.min_count))
        logger.info('reset weights')

        if self.hybrid_pred:
            freq_list = [self.vocab[v].count for v in self.vocab if len(v)==1]
            freq_list.sort(reverse=True)
            self.hybrid_threshold = freq_list[len(freq_list)/25]
            print '>frequencey threshold for hybrid prediction is:', self.hybrid_threshold

        self.reset_weights()


    def reset_weights(self):

        logger.info("resetting layer weights")
        self.syn0 = empty((len(self.vocab), self.layer1_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in xrange(len(self.vocab)):
            random.seed(uint32(self.hashfxn(self.index2word[i] + str(self.seed))))
            if self.pre_train:
                word = self.index2word[i]
                if word in self.uni_emb and not np_isnan(self.uni_emb[word]).any():
                    self.syn0[i] = np_copy(self.uni_emb.syn1neg[self.uni_emb.vocab[word].index])
                    #self.syn0[i] = np_copy(self.uni_emb[word])
                #    print '##word', word, 'vec=', self.syn0[i]
                elif word in self.bi_emb and not np_isnan(self.bi_emb[word]).any():
                    self.syn0[i] =  np_copy(self.bi_emb.syn1neg[self.bi_emb.vocab[word].index])
                    #self.syn0[i] = np_copy(self.bi_emb[word])

                else:
                    self.syn0[i] = (random.rand(self.layer1_size) - 0.5) / self.layer1_size
            else:
                self.syn0[i] = (random.rand(self.layer1_size) - 0.5) / self.layer1_size


        self.syn1neg = zeros((len(self.vocab), self.pred_size), dtype=REAL)



    def _prepare_sentences(self, sentences):
        for sentence in sentences:
            yield sentence  #"sample" every word itself, rather than their vocab objects



    def _get_job_words(self, alpha, work, job, neu1):
        total_count, total_error_count = 0, 0
        for sentence in job:
            x,y = self.train_gold_per_sentence(sentence, alpha, work)
            total_count +=x
            total_error_count += y

        print '\n===> batch subgram error rate =', total_error_count/float(total_count),\
            'subgram_error/subgram count=', total_error_count, total_count,'\n'

        if random.random()>0.85:
            print '\n==='
            self.train_mode = False
            f_score, oov_recall, iv_recall = self.eval(self.quick_test_corpus, self.quick_test)
            print 'Quick segEval: F-score/OOV-Recall/IV-recall=', f_score, oov_recall, iv_recall
            self.train_mode= True

        return total_count




    def __str__(self):
        return "Subword2Vec(vocab=%s, size=%s, alpha=%s)" % (len(self.index2word), self.layer1_size, self.alpha)

    def save(self, *args, **kwargs):
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm'])  # don't bother storing the cached normalized vectors
        super(Seger, self).save(*args, **kwargs)




if __name__=='__main__':
    pass



