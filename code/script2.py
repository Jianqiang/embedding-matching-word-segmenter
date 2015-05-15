# -*- coding: utf-8 -*-
'''
script for running seg2.py (greedy segmenter based on embedding matching)
use the config.txt as the argument of this script to reproduce the ACL paper results
'''

from seg2 import Seger
import codecs,logging, time, os, datetime, sys

def parse_config(param_table, path_config):

    #print '###>>>', param_table
    with codecs.open(path_config,'rU','utf-8') as f:
        for line in f:
            if not line.strip() or line.strip()[0]=='#':
                pass

            else:
                tokens=line.strip().split('=')
                if len(tokens)==2:
                    key=tokens[0].strip()
                    value = tokens[1].strip()
                    if key in param_table:
                        if key in {'pre_train','hybrid_pred', 'no_bigram_feature','no_action_feature','no_unigram_feature','no_binary_action_feature','no_sb_state_feature'}:
                            if value.lower() == 'false':
                                param_table[key]=False
                            elif value.lower() == 'true':
                                param_table[key]=True
                            else:
                                print 'Error! the value for',key, ' should be either True or False (case-insensitive)'
                                assert False


                        elif key=='iter':
                            param_table[key]=int(value)

                        elif key =='alpha':
                            param_table[key] = float(value)

                        elif key in {'train_path','test_raw_path', 'test_path', 'dev_path','quick_test','dict_path',
                                     'score_script_path','uni_path','bi_path'}:

                            if os.path.isfile(value):
                                param_table[key]=value
                            else:
                                print 'Error! the file for ', key, ',', value, 'is not a valid file! Param parse fail!'
                                assert False

                        elif key =='model_path':
                            if  os.path.isdir(os.path.dirname(value)):
                                param_table[key] = value

                            else:
                                print 'Error! the dir specified by the model path', value, 'does NOT exit!'
                                assert False

                    else:
                        print 'Error Unknown param name:', key, 'program exit!'
                        assert False

                else:
                    print 'warning: current line is NOT in the format of xxx=zzz'



    return param_table




if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    start = time.time()

    model_path = './model.'+datetime.datetime.now().strftime('%d.%H.%M')+'.new'
    param_table={'iter':1,'alpha':0.1,'model_path':model_path,\
                 'train_path':None, 'test_raw_path': None, 'test_path': None, 'dev_path':None, 'quick_test': None,
                 'dict_path': None, 'score_script_path':None, 'pre_train':False , 'uni_path':None, 'bi_path':None, 'hybrid_pred':False,
    'no_action_feature':False, 'no_bigram_feature':False, 'no_unigram_feature':False,  'no_binary_action_feature':False, \
    'no_sb_state_feature':False }

    print '\n>> parse parameters...'

    table = parse_config(param_table, sys.argv[1])

    print '\n=====parameters===='
    for k in sorted(table.keys()):
        print '\t',k, '=', table[k]

    print '\n>> initial model...'
    seger=Seger (workers=1, iter=table['iter'], alpha=table['alpha'],
                 train_path= table['train_path'],  test_raw_path = table['test_raw_path'], test_path = table['test_path'],
                 dev_path = table['dev_path'], quick_test = table['quick_test'], dict_path= table['dict_path'],
                 score_script_path = table['score_script_path'], pre_train = table['pre_train'] ,\
                 uni_path=table['uni_path'], bi_path=table['bi_path'], hybrid_pred=table['hybrid_pred'],\
                 no_action_feature = table['no_action_feature'], no_binary_action_feature=table['no_binary_action_feature'],
                 no_bigram_feature=table['no_bigram_feature'], no_unigram_feature=table['no_unigram_feature'], no_sb_state_feature=table['no_sb_state_feature'])


    print '\n>> do training...'
    seger.do_training()
    seger.save(model_path)
    print 'done in ', time.time()-start





