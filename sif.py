import sys
import os

import numpy as np

sys.path.append('SIF/src')
import SIF_embedding, params, data_io

"""
1. Initialize sentence embedding using the SIF algorithm over training data

word_weights : a / (a + p(w)) - calculate unigram probabilities over training data
"""

def get_word_weights(word_freq_file, a=1e-3):
    word_weights = {}
    N = 0

    with open(word_freq_file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                line = line.split()
                if len(line) == 2:
                    word_weights[line[0]] = float(line[1])
                    N += float(line[1])
                else:
                    print(line)

    for key, value in word_weights.items():
        word_weights[key] = a / (a + value / N)

    return word_weights

def load_weights():
    if os.path.isfile('word_weights.npy'):
        weights = np.load('word_weights.npy', allow_pickle=False).squeeze()
    else:
        word_weights = get_word_weights('SIF/auxiliary_data/enwiki_vocab_min200.txt')
        # print(type(word_weights))
        #print(word_weights.items()[:5])

        # create numpy matrix of weights using word2ix
        weights = np.zeros((max(word2ix.values()) + 1))
        unk = 0
        for word, ix in word2ix.items():
            if word.lower() not in word_weights.keys():
                weights[ix] = 1.
                unk += 1
            else:
                weights[ix] = word_weights[word.lower()]
        # for word, ix in word2ix.items():
        print("# of words with unknown weight", unk)
        print(weights[:5])

        np.save('word_weights.npy', weights, allow_pickle=False)
    return weights

def get_sentence_word_weights(text, weights):
    return data_io.seq2weight(text, np.ones(text.shape), weights)

# get weights for each word in each sentence
# train_w = data_io.seq2weight(train['text'], np.ones(train['text'].shape), weights)
# valid_w = data_io.seq2weight(valid['text'], np.ones(valid['text'].shape), weights)
# test_w = data_io.seq2weight(test['text'], np.ones(test['text'].shape), weights)
# 
# weights = torch.tensor(weights, device=device, dtype=torch.float32)
# word_embeddings = torch.tensor(word_embeddings, device=device, dtype=torch.float32)

# normalize word embedding lengths
# print(word_embeddings.norm(dim=-1).max())
# print('embed_size', word_embeddings.size())
# word_embeddings = F.normalize(word_embeddings)
# print(word_embeddings.norm(dim=-1).max())

#print(word_weights.keys()[:10])
def get_word_embeddings(word_embeddings, weights, text):
    text_w = get_sentence_word_weights(text, weights)

    # number of principal components to remove
    RMPC = 1
    p = params.params()
    p.rmpc = RMPC

    embedding = SIF_embedding.SIF_embedding(word_embeddings, text, text_w, p)

    return embedding
