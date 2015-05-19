#Embedding Matching for Chinese Word Segmentation

This repository hosts the code for the following paper:

Ma and Hinrichs. 2015. Accurate Linear-Time Chinese Word Segmentation via Embedding Matching. In *Proceedings of ACL*.


The python implementation is in the /code folder. Change the directory there to get started.

##Usage:
1. To reproduce the experiments, including the training and evaluations after each iteration, run:

python  script2.py  config.txt

the script2.py parses the configurations in config.txt and calls the main code seg2.py to train the model.

2. To segment a raw corpus with an existing model

python predict  <model> <input_corpus> <output_path>

Note: Python2 is supported and all corpora are assumed to be in UTF-8 encoding.

* The training takes about 1 hour on a laptop with intel Core5 CPU (single thread). In the process, error rate and quick evaulations for small batches of sentences will be conducted. 

At the end of each epoch, evaluation on the testing data is given. And in the end of the whole training (10 epochs), F-scores, OOV-Recall and IV recall for each epoch will be reported again. Evaluations are based on the SIGHAN official script (in "working_data" folder)

##Requriement:
###Required Software:
- Linux-like environment  (tested on Ubuntu 14.4 and Mac OSX )
- python 2.7 or above (but not Python 3.X)
- Python numpy package 1.9 or above
- Python gensim package 0.10.3 (we use their save/load/math routines and also the threading, lookup-table etc from their word2vec implementation, which is modified a bit and included here)


### Required Data
PKU corpus from 2nd SIGHAN word segmentation bakeoff (http://www.sighan.org/bakeoff2005/) is in the folder "working_data", which include:

- score (official scoring script: http://www.sighan.org/bakeoff2003/score)
- pku_test (official testing split)
- pku_train (official training split)
- pku_test.raw (testing data converted into character sequences)
- pku.100.txt (top 100 sent in pku_train, for fast evaluation when monitoring training)
- pku.test.10 (top10 sent in pku_test, similar purpose)
- pku.dict (the list of words that have occurred in pku_train, for evaluation)

MSR and other dataset can be prepared in similar manner (.raw, .100.txt test.10, and .dict files are needed) and update config.txt accordingly.

------------- 
This software is open source software released under the GNU LGPL license. Copyright (c) 2015-now Jianqiang Ma
