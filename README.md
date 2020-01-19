# Word-Embedding-Generator

Word-Embedding-Generator is a project that generates word embeddings from raw texts. It supports Word2Vec, ELMO and BERT models for word embedding pre-training.

## Getting Started
### Dependencies
-   Python 3.7
-   NumPy
-   PyTorch
-   NLTK
-   Gensim
#### Data Ready
#### Raw Data
Put the raw text data to be trained at a folder path. The program will automatically scan all the files in the path.

#### Pre-trained Models

Please download the pre-trained word embedding models here:

Google Word2Vec: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

#### Other Files

please add stopwords in this file: `./stopwords` 

### Training Word Embedding

```bash
python create_embb.py --load_path ./raw_data/ --save_path ./  --lower --freq 1 --dim 300 --model_path ./word2vec.model
```
- Set `--freq` to k to ignore the words with frequencies less than k
- Set `--size` for the maximum vocab size, default=500000
- Set `--dim` for the word embedding dimension, default=100
- Leave `--model_path` blank to train Word2Vec model from scratch using the raw data
- Set `--model_path` to 'google' to use pre-trained Google Word2Vec model 
- Set `--model_path` to the saved model path to load the pre-trained Word2Vec model 

The program will save two files, `vocab.pkl` and `embb.pkl` for word to index (and vice versa) and index to word embedding translation respectively.

### Loading Word Embedding

```bash
python load_embb.py --vocab_path ./vocab300.pkl --embb_path ./embb300.pkl
```
You may see comments in this file for the detailed usages.

## TODO
ELMO and BERT model support will be added by the end of Jan 2020. <br>
If you have ANY difficulties to get things working in the above steps, feel free to open an issue. You can expect a reply within 24 hours.
