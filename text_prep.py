import unicodedata
import string
import torch
import torch.nn as nn
import gensim
from gensim.models.word2vec import Word2Vec

w2v_path = "C:/Users/blackbak/Documents/github/data/squad_data/GoogleNews-vectors-negative300.bin.gz"

all_letters = string.ascii_letters + " '" # + " ?.,;'" remove everything that is not a letter
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

###pad each sentence with <\s> as eos token t
def sentence2idx(model, sentence):
    sentence = unicodeToAscii(sentence)
    sentence = sentence.lower()
    idx_list = [word2idx(model, word) for word in sentence.split()]
    idx_list = [i for i in idx_list if i is not None]
    idx_list.append(word2idx(model, '</s>'))
    return idx_list

################ gensim model helper functions
def vector2word(model, vector):
    idx = cosine_similarity(model.wv.vectors, vector)
    word = idx2word(model, idx)
    return word

def cosine_similarity(input1, input2):
    ###input1 is the matrix
    ###input2 is the vector
    input1 = input1.to(device)
    input2 = input2.to(device)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    output = cos(input1, input2)
    return torch.argmax(output)

def idx2word(model, index):
    ### model is a gensim keyed model
    return model.wv.index2word[index]

def word2idx(model, word):
    ### model is a gensim keyed model
    try:
        idx = model.wv.vocab[word].index
        return idx
    except:
        pass
    
def idx2vector(model, idx):
    return model.wv.vectors[idx, :]
    
def gensim2embedding(model, device):
    weights = torch.FloatTensor(model.wv.vectors).to(device)
    embedding = nn.Embedding.from_pretrained(weights, freeze=True)
    return embedding

def build_w2v_model(sentence_list):
    sentence_list = [unicodeToAscii(sentence) for sentence in sentence_list]
    tokenized_list = [sentence.lower().split() for sentence in sentence_list]
    model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    model_new = Word2Vec(size=300, min_count = 0)
    model_new.build_vocab(tokenized_list)
    total_examples = model_new.corpus_count
    model_new.build_vocab([list(model.vocab.keys())], update=True)
    model_new.intersect_word2vec_format(w2v_path, binary=True, lockf=0.0)
    model_new.train(tokenized_list, total_examples=total_examples, epochs=model_new.epochs)
    return model_new
    