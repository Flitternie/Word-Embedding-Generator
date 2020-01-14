import codecs
import torch
from collections import OrderedDict 

class Dict(object):
    def __init__(self, data=None, lower=None):
        self.idxToWord = {}
        self.wordToIndex = {}
        self.frequencies = {}
        self.lower = {}

        if data is not None:
            if type(data) == str:
                self.loadFile(data)
    
    def size(self):
        return len(self.idxToWord)
    
    def loadFile(self, filename):
        for line in codecs.open(filename,'r','utf-8'):
            fields = line.split()
            word = fields[0]
            idx = int(fields[1])
            self.add(word, idx)
    
    def writeFile(self, filename):
        with open(filename, 'w') as file:
            for i in range(self.size()):
                word = self.idxToWord[i]
                file.write('%s %d\n' % (word, i))
        file.close()
    
    def loadDict(self, idxToWord):
        for i in range(len(idxToWord)):
            label = idxToWord[i]
            self.add(label, i)

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.wordToIndex[key]
        except KeyError:
            return default

    def getLabel(self, idx, default=None):
        try:
            return self.idxToWord[idx]
        except KeyError:
            return default

    def add(self, word, idx=None):
        word = word.lower() if self.lower else word
        if idx is not None:
            self.idxToWord[idx] = word
            self.wordToIndex[word] = idx
        else:
            if word in self.wordToIndex:
                idx = self.wordToIndex[word]
            else:
                idx = len(self.idxToWord)
                self.idxToWord[idx] = word
                self.wordToIndex[word] = idx
        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1
        return idx

    def prune(self, size, freq):
        if size > self.size():
            idx = []
            for i in range(len(self.frequencies)):
                if self.frequencies[i] > freq:
                    idx.append(i)
            newDict = Dict()
            for i in idx:
                newDict.add(self.idxToWord[i])
            return newDict

        # Only keep the `size` most frequent entries.
        freq_list = []
        for i in range(len(self.frequencies)):
            if self.frequencies[i] > freq:
                freq_list.append(self.frequencies[i])
        freq = torch.tensor(freq_list)
        _, idx = torch.sort(freq, 0, True)
        idx = idx.tolist()

        newDict = Dict()
        newDict.lower = self.lower

        for i in idx[:size]:
            newDict.add(self.idxToWord[i])

        return newDict

    def convertToIdxandOOV(self, words, unk='<unk>'):
        vec = []
        oovs = OrderedDict()
        for word in words:
            id = self.lookup(word, default=unk)
            if id != unk:
                vec += [id]
            else:
                if word not in oovs:
                    oovs[word] = len(oovs)+self.size() 
                oov_num = oovs[word]
                vec += [oov_num]
        return torch.LongTensor(vec), oovs

    def convertToWords(self, idx, oovs=None):
        words = []
        for i in idx:
            if i < self.size():
                words += [self.getLabel(i)]
            else:
                words.append(list(oovs.items())[i-self.size()][0])
        return words