import argparse
import pickle
import os
import string
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import RegexpTokenizer
import numpy as np

import utils
import gensim.models

parser = argparse.ArgumentParser(description='create_embb.py')
parser.add_argument('--load_path', required=True,
                    help="input path for the data")
parser.add_argument('--save_path', required=True,
                    help="output path for the prepared data")
parser.add_argument('--model_path',
                    help='path for the w2v model, leave blank to train from scratch')
parser.add_argument('--size', type=int, default=500000,
                    help="maximum size of the vocabulary")
parser.add_argument('--dim', type=int, default=100,
                    help="dimension of the word embedding")
parser.add_argument('--lower', action='store_true',
                    help='lower case')
parser.add_argument('--freq', type=int, default=3,
                    help="remove words less frequent")

opt = parser.parse_args()
google = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)

def readRaw(files, lower=True):
    # nltk.download('punkt')
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    with open("./stopwords", encoding='latin-1') as F:
        stopWords = set(map(str.strip, F.readlines()))
    h_tokens_list  = []
    tokens_list    = []
    for file in files:
        with open(file, encoding='latin-1') as f:
            for sent in f.readlines():
                if lower: 
                    sent = sent.strip().lower()
                else: 
                    sent = sent.strip()
                sentences = sent_tokenizer.tokenize(sent)
                h_tokens = []
                tokens   = []
                for sentence in sentences:
                    delEStr = string.punctuation + string.digits
                    words = tokenizer.tokenize(str(sentence))
                    symbols = list(string.punctuation + string.digits)
                    symbols.remove('!')
                    elements = words
                    words = []
                    for word in elements:
                        if word not in symbols:
                            if word != '!':
                                word = word.translate({ord(i): None for i in delEStr})
                            if len(word) != 0:
                                words.append(word)

                    if len(words) > 0:
                        if len(words) == 1 and (words[0] == '!' or words[0] in stopWords):
                            pass
                        else:
                            h_tokens.append(words)
                            tokens.extend(words)
                h_tokens_list.append(h_tokens)
                tokens_list.append(tokens)
    # print(np.array(h_tokens_list).shape, np.array(tokens_list).shape)
    return h_tokens_list, tokens_list

def makeVocab(files, size, vocab, freq, lower=True):
    _, sentences = readRaw(files, lower)
    for sentence in sentences:
        for word in sentence:
            vocab.add(word)
    if size > 0:
        originalSize = vocab.size()
        vocab = vocab.prune(size, freq)
        print('Created dictionary of size %d (pruned from %d)' %
              (vocab.size(), originalSize))
    return vocab

def saveVocab(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    with open(file, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

def trainW2V(files, freq, dim):
    print("Training Word2Vec model. This may take a while.")
    _, sentences = readRaw(files)
    print(len(sentences))
    model = gensim.models.Word2Vec(sentences=sentences, min_count=0, size=dim, workers=4)
    model.save(os.path.join(opt.save_path, "word2vec%d.model"%(dim)))
    return model

def makeEmbb(vocab, model, dim):
    embb = []
    for key, value in vocab.idxToWord.items():
        if opt.model_path != 'google':
            try:
                embb.append(google[value])
            except KeyError:
                embb.append(model[value])
        else:
            try:
                embb.append(model[value])
            except KeyError:
                embb.append(np.ones(dim))
    embb = np.array(embb) 
    try:
        assert embb.size == vocab.size() * dim
    except KeyError:
        print("Size not matching")
    return embb

def main():
    dicts={}
    embbs={}
    domains = ['books', 'dvd', 'electronics', 'kitchen', 'video']
    files = []
    for r,d,f in os.walk(opt.load_path):
        for file in f:
            files.append(os.path.join(r, file))
    print("Done reading all files")
    if opt.model_path == None:
        model = trainW2V(files, freq=opt.freq, dim=opt.dim)
    elif opt.model_path == "google":
        model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)
    else:
        model = gensim.models.Word2Vec.load(opt.model_path)
    print("Done training/loading the W2V")
    print(model.wv.vectors.shape)
    for src_domain in domains:
        dicts[src_domain] = {}
        embbs[src_domain] = {}
        for tgt_domain in domains:        
            if src_domain != tgt_domain:
                print("Start handling the files for %s to %s" %(src_domain, tgt_domain))
                files = []
                for r,d,f in os.walk(os.path.join(opt.load_path, src_domain)):
                    for file in f:
                        files.append(os.path.join(r, file))
                for r,d,f in os.walk(os.path.join(opt.load_path, tgt_domain)):
                    for file in f:
                        files.append(os.path.join(r, file))
                dicts[src_domain][tgt_domain] = makeVocab(files, size=opt.size, vocab=utils.Dict(None, lower=opt.lower), freq=opt.freq, lower=opt.lower)
                embbs[src_domain][tgt_domain] = makeEmbb(dicts[src_domain][tgt_domain], model, dim=opt.dim)
                print("Done handling the files for %s to %s" %(src_domain, tgt_domain))

    for key, value in dicts.items():
        print(key, value)
    for key, value in embbs.items():
        print(key, value)

    saveVocab('dict', dicts, os.path.join(opt.save_path, "vocab%d.pkl"%(opt.dim)))
    saveVocab('embb', embbs, os.path.join(opt.save_path, "embb%d.pkl"%(opt.dim)))
                

if __name__ == "__main__":
    main()
