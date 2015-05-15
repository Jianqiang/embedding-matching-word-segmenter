# -*- coding: utf-8 -*-
import sys, datetime, time, codecs
from seg2 import *


def segment_corpus(model, corpus, threashold=0):
    tic = time.time()
    count = 0
    seg_corpus = []
    for sent_no, sent in enumerate(corpus):
        if not sent_no%100:
            print 'num of sentence segmented:', sent_no, '...'

        tokens = []
        if sent:
            old_sentence = "".join(sent)
            sentence = map(full2halfwidth, old_sentence) # all half-width version, used to predict label...

            prev2_label, prev_label = 0, 0


            for pos, char in enumerate(old_sentence): # char is still the char from original sentence, for correct eval
                if pos== 0:
                    label = 0
                else:
                    score_list, _, _, _, _ = model.predict_sigle_position(sentence, pos, prev2_label, prev_label)

                    if model.binary_pred:
                        score_list = score_list[:2]

                    elif model.alter:
                        old_char = old_sentence[pos]
                        if old_char in model.vocab and model.vocab[old_char].count>threashold:
                            score_list = score_list[-2:]
                        else:
                            #score_list = score_list[:2]
                            x,y= score_list[:2], score_list[-2:]
                            score_list = [(x[i]+y[i])/2.0 for i in range(2)]

                    elif model.hybrid_pred:
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
            count += (pos+1)
        seg_corpus.append(tokens)

    diff = time.time() - tic
    print 'segmentation done!'
    print 'time spent:', diff, 'speed=', count/float(diff), 'characters per second'

    return seg_corpus



if __name__ == '__main__':

    print '\n\nScript for conducting segmentation with an existing model...'
    print '\nArg: 1. model_path, 2. file_to_be_segmented,  3. path to output'


    
    model_path, test_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    print '\nreading testing corpus...'
    test_corpus = [''.join(line.split()) for line in codecs.open(test_path, 'rU','utf-8')]


    print '\nloading model...'
    model = Seger.load(model_path)
    model.drop_out=False
    model.alter = True
    threshold = model.hybrid_threshold
    seged = segment_corpus(model, test_corpus, threshold)
    print '\nwriting segmented corpus to file', out_path

    with codecs.open(out_path, 'w','utf-8') as f:
        for sent in seged:
            f.write(' '.join(sent)+'\n')

    print 'written done!' 



