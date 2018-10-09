import os
import json
import time
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import numpy as np

import data_loader as loader
from losses import get_log_prob_matrix
from sentiment_model import train_sentiment, eval_sentiment, train_sentiment_for_latents

sys.path.append('SIF/src')
import SIF_embedding, params, data_io

torch.cuda.set_device(3)
device = torch.device('cuda')

"""
Hyper-parameters:
- Batch size
- Hidden size for sentiment model
- Learning rate
- Sentiment learning rate
- Segment len
- use dot-prod or angular distance
- # of epochs to optimize embeddings
- Type of optimizer?
"""
def read_config(config_file):
    # Future work: handle default values?
    return json.load(open(config_file, 'r'))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--sentiment_hidden_size', type=int, default=100)
    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--sentiment_lr', type=float, default=1e-2)
    # parser.add_argument('--seq_len', type=int, default=20)
    # parser.add_argument('--word_sim_metric', choices=['angular, dot_prod'], default='angular')
    # parser.add_argument('--n_epochs', type=int, default=100)
    # parser.add_argument('--n_sentiment_epochs', type=int, default=100)

    args = vars(parser.parse_args())
    config = read_config(args['config_file'])
    print('######################################')
    print("Config: {}".format(config['config_num']))
    args.update(config)

    return args

args = parse_arguments()

"""
Procedure:
1. Initialize sentence embedding using the SIF algorithm
2. Initialize linear regression model to mean and variance (element-wise independent)
    of audio and visual features
3. Maximize joint log-probability of generating sentence, audio, and visual features.
"""

"""
Load data

max_segment_len can be varied to compare results?

Note: loader.load_word_level_features always returns the same split.
"""
MAX_SEGMENT_LEN = args['seq_len']
TR_PROPORTION = 2.0/3
# number of principal components to remove
RMPC = 1

def load_data(max_len, tr_proportion):
    word2ix = loader.load_word2ix()

    # load glove 300d word embeddings
    word_embeddings = loader.load_word_embedding()
    
    # TODO: save train, valid, test in file so that don't have to
    # regenerate each time
    train, valid, test = loader.load_word_level_features(max_len, tr_proportion)

    return word2ix, word_embeddings, (train, valid, test)

word2ix, word_embeddings, data = load_data(MAX_SEGMENT_LEN, TR_PROPORTION)
train, valid, test = data

"""
Combine data:

Since we are comparing performance of the sentiment model before we add in audio and visual,
and after, we should optimize all embeddings at once.

But for sentiment, we should still keep the same train/valid/test split.

"""

def normalize_data(train):
    """
    normalize audio and visual features to [-1, 1].
    Also remove any features that are always the same value.
    """
    # normalize audio and visual features to [-1, 1]
    audio_min = train['covarep'].min((0, 1))
    audio_max = train['covarep'].max((0, 1))
    # print(audio_max - audio_min)
    audio_diff = audio_max - audio_min
    audio_nonzeros = (audio_diff == 0).nonzero()[0]
    # print(audio_nonzeros)
    audio_nonzeros = audio_diff.nonzero()[0]
    # print(train['covarep'].shape)
    # print(train['covarep'][:, :, audio_nonzeros].shape)

    train['covarep'] = train['covarep'][:, :, audio_nonzeros]

    audio_min = train['covarep'].min((0, 1))
    audio_max = train['covarep'].max((0, 1))
    audio_diff = audio_max - audio_min

    vis_min = train['facet'].min((0, 1))
    vis_max = train['facet'].max((0, 1))

    train['covarep'] = (train['covarep'] + audio_min) * 2. / (audio_max - audio_min) - 1.
    train['facet'] = (train['facet'] + vis_min) * 2. / (vis_max - vis_min) - 1.

    return train

train = normalize_data(train)
valid = normalize_data(valid)
test = normalize_data(test)

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

# get weights for each word in each sentence
train_w = data_io.seq2weight(train['text'], np.ones(train['text'].shape), weights)
valid_w = data_io.seq2weight(valid['text'], np.ones(valid['text'].shape), weights)
test_w = data_io.seq2weight(test['text'], np.ones(test['text'].shape), weights)

weights = torch.tensor(weights, device=device, dtype=torch.float32)
word_embeddings = torch.tensor(word_embeddings, device=device, dtype=torch.float32)

# normalize word embedding lengths
# print(word_embeddings.norm(dim=-1).max())
# print('embed_size', word_embeddings.size())
# word_embeddings = F.normalize(word_embeddings)
# print(word_embeddings.norm(dim=-1).max())

#print(word_weights.keys()[:10])
params = params.params()
params.rmpc = RMPC

train_embedding = SIF_embedding.SIF_embedding(word_embeddings, train['text'], train_w, params)
valid_embedding = SIF_embedding.SIF_embedding(word_embeddings, valid['text'], valid_w, params)
test_embedding = SIF_embedding.SIF_embedding(word_embeddings, test['text'], test_w, params)

class MMData(Dataset):
    def __init__(self, text, audio, visual, device):
        super(Dataset, self).__init__()
        
        if not torch.is_tensor(text):
            text = torch.tensor(text, device=device, dtype=torch.long)
        if not torch.is_tensor(audio):
            audio = torch.tensor(audio, device=device, dtype=torch.float32)
        if not torch.is_tensor(visual):
            visual = torch.tensor(visual, device=device, dtype=torch.float32)

        assert text.size()[0] == audio.size()[0]
        assert audio.size()[0] == visual.size()[0]

        self.text = text
        self.audio = audio
        self.visual = visual
        self.len = self.text.size()[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return idx, self.text[idx], self.audio[idx], self.visual[idx]

class SentimentData(Dataset):
    def __init__(self, sentiment, device):
        super(Dataset, self).__init__()

        if not torch.is_tensor(sentiment):
            sentiment = torch.tensor(sentiment, device=device, dtype=torch.float32)

        self.sentiment = sentiment

    def __len__(self):
        return self.sentiment.size()[0]

    def __getitem__(self, idx):
        return idx, self.sentiment[idx]

BATCH_SIZE = args['batch_size']
dataset = MMData(train['text'], train['covarep'], train['facet'], device)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
senti_dataset = SentimentData(train['label'], device)
# print(len(senti_dataset))
senti_dataloader = DataLoader(senti_dataset, batch_size=BATCH_SIZE, shuffle=True)

"""
2. Initialize regression model to generate mean and variance of audio and
    visual features
"""

class AudioVisualGenerator(nn.Module):
    def __init__(self, embedding_dim, audio_dim, visual_dim, frozen_weights=True):
        super(AudioVisualGenerator, self).__init__()

        self.embedding = None
        self.embedding_dim = embedding_dim

        self.embed2audio_mu = nn.Linear(self.embedding_dim, audio_dim)
        self.embed2audio_sigma = nn.Linear(self.embedding_dim, audio_dim)

        self.embed2visual_mu = nn.Linear(self.embedding_dim, visual_dim)
        self.embed2visual_sigma = nn.Linear(self.embedding_dim, visual_dim)

        if frozen_weights:
            self.freeze_weights()

    def freeze_weights(self):
        # freeze weights
        for param in self.embed2audio_mu.parameters():
            param.requires_grad = False
        for param in self.embed2audio_sigma.parameters():
            param.requires_grad = False
        for param in self.embed2visual_mu.parameters():
            param.requires_grad = False
        for param in self.embed2visual_sigma.parameters():
            param.requires_grad = False

    def init_embedding(self, embedding):
        assert embedding.size()[-1] == self.embedding_dim

        self.embedding = embedding
        self.embedding.requires_grad = True
        self.embedding_dim = self.embedding.size()[-1]

    def forward(self, embeddings):
        #assert self.embedding is not None

        #to_gen = self.embedding[idxes]
        to_gen = embeddings

        # from sentence embedding, generate mean and variance of
        # audio and visual features
        # since variance is positive, we exponentiate.
        audio_mu = self.embed2audio_mu(to_gen)
        audio_sigma = self.embed2audio_sigma(to_gen).exp()

        visual_mu = self.embed2visual_mu(to_gen)
        visual_sigma = self.embed2visual_sigma(to_gen).exp()

        return (audio_mu, audio_sigma), (visual_mu, visual_sigma)

EMBEDDING_DIM = word_embeddings.size()[-1]
AUDIO_DIM = train['covarep'].shape[-1]
VISUAL_DIM = train['facet'].shape[-1]

#optimizer = optim.SGD(gen_model.parameters(), lr=0.01, momentum=0.9)

"""
3. Maximize log-probability over all modalities

Each epoch, generate mean and variance using linear regression.
Calculate log-probability of generated audio and visual features
(assume independence)

Also calculate log-probability of word embeddings, using model in Arora paper
(with either softmax of inner product, or angular distance)

Then minimize negative log-probability using gradient descent.
"""

SENTIMENT_HIDDEN_DIM = args['sentiment_hidden_size']

valid_niter = 10

# sentiment analysis
curr_embedding = torch.tensor(train_embedding.copy(), device=device, dtype=torch.float32)

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("Initial sentiment predictions, before optimizing audio and visual")
train_sentiment_for_latents(args, curr_embedding, senti_dataloader, device)

gen_model = AudioVisualGenerator(EMBEDDING_DIM, AUDIO_DIM, VISUAL_DIM, frozen_weights=True).to(device)

print("Training...")

curr_embedding.requires_grad = True

lr = args['lr']
optimizer = optim.SGD([curr_embedding], lr=lr)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)

N_EPOCHS = args['n_epochs']
start_time = time.time()
for i in range(N_EPOCHS):
    epoch_loss = 0.
    iters = 0
    #curr_embedding = F.normalize(curr_embedding)
    for j, text, aud, vis in dataloader:
        iters += 1
        optimizer.zero_grad()
        # print(curr_embedding[:10])
        audio, visual = gen_model(curr_embedding[j])

        if audio[1].min().abs() < 1e-7:
            print("boo!")
        if visual[1].min().abs() < 1e-7:
            print("boot!")

        log_prob = -get_log_prob_matrix(args, curr_embedding[j], audio, visual,
                {"text": text, "covarep": aud, "facet": vis}, word_embeddings, weights, device=device)

        avg_log_prob = log_prob.mean()
        avg_log_prob.backward(retain_graph=True)

        # nn.utils.clip_grad_norm_([gen_model.embedding], 500)

        optimizer.step()
        epoch_loss += avg_log_prob
    # scheduler.step()
    if i % valid_niter == 0:
        print("epoch {}: {} ({}s)".format(i, epoch_loss / iters, time.time() - start_time))
curr_embedding.requires_grad = False

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("Initial sentiment predictions, AFTER optimizing audio and visual")
train_sentiment_for_latents(args, curr_embedding, senti_dataloader, device)

sys.stdout.flush()

