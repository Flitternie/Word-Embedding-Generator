import argparse
import pickle
import os
import string
import numpy as np
import torch
import utils

parser = argparse.ArgumentParser(description='load_embb.py')
parser.add_argument('--vocab_path', required=True,
                    help="input path for the vocab")
parser.add_argument('--embb_path', required=True,
                    help="input path for the embbedings")
                    

opt = parser.parse_args()

# load dict and embb from files
def loadFiles(dicts=os.path.join(opt.vocab_path), embbs=os.path.join(opt.embb_path)):
    with open(dicts, 'rb') as f:
        dicts = pickle.load(f)
    with open(embbs, 'rb') as f:
        embbs = pickle.load(f)
    return dicts, embbs

# specify source and target domain
def loadDict(src, tgt):
    domains = ['books', 'dvd', 'electronics', 'kitchen', 'video']
    assert src in domains and tgt in domains
    dicts, embbs = loadFiles()
    dic = dicts[src][tgt]
    embb = embbs[src][tgt]
    return dic, embb

# translate a list of words to a list of index with specified dict, save OOVs to another dict
def wordToIdx(words, dic):
    idx, oov = dic.convertToIdxandOOV(words)
    return idx, oov

# translate a list of index to a list of words with specified vocab dict and OOV dict
def idxToWord(idxs, dic, oov=None):
    return dic.convertToWords(idxs.tolist(), oov)

# translate a list of index to a word embedding tensor with specified embb, assign 1s to OOV words
def idxToEmbb(idxs, embb):
    vec = []
    for i in idxs:
        if i < embb.shape[0]:
            vec.append(embb[i])
        else:
            vec.append(np.ones(embb.shape[1]))
    return torch.LongTensor(vec)
        

def main():
    demo = 'This is a demo of aigoster! How is it?'    # a demo of input format
    for char in string.punctuation:                         # delete all special symbols
        demo = demo.replace(char, '')
    dic, embb = loadDict('books', 'dvd')                    # load vocab and embb prepared for src and tgt domain
    idx, oov = wordToIdx(demo.lower().split(), dic)                 # translate word to index, save oov words to another dict
    embb = idxToEmbb(idx, embb)                             # translate index to embedding, return a torch tensor
    words = idxToWord(idx, dic, oov)                        # translate index to word with oov dict
    print(words)

    

if __name__ == "__main__":
    main()