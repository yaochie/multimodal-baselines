import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

import data_loader as loader

import os
import time
import sys
sys.path.append('SIF/src')
import SIF_embedding, params, data_io

torch.cuda.set_device(2)
device = torch.device('cuda')

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
"""
MAX_SEGMENT_LEN = 20
TR_PROPORTION = 2.0/3
# number of principal components to remove
RMPC = 1

word2ix = loader.load_word2ix()
# load glove 300d word embeddings
word_embeddings = loader.load_word_embedding()
train, valid, test = loader.load_word_level_features(MAX_SEGMENT_LEN, TR_PROPORTION)

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

#print(word_embeddings.shape)
#print(train['text'].shape)
#print(type(word2ix))
#print(len(word2ix))
#print(word2ix.items()[:10])
#print(max(word2ix.values()))
#from collections import Counter
#c = Counter()
#for v in word2ix.values():
#    c[v] += 1
#print(sorted(c.values(), reverse=True)[:10])

if os.path.isfile('word_weights.npy'):
    weights = np.load('word_weights.npy', allow_pickle=False).squeeze()
else:
    word_weights = get_word_weights('SIF/auxiliary_data/enwiki_vocab_min200.txt')
    print(type(word_weights))
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

BATCH_SIZE = 16
dataset = MMData(train['text'], train['covarep'], train['facet'], device)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
senti_dataset = SentimentData(train['label'], device)
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

    def forward(self, idxes):
        assert self.embedding is not None

        to_gen = self.embedding[idxes]

        # from sentence embedding, generate mean and variance of
        # audio and visual features
        # since variance is positive, we exponentiate.
        audio_mu = self.embed2audio_mu(to_gen)
        audio_sigma = self.embed2audio_sigma(to_gen).exp()

        visual_mu = self.embed2visual_mu(to_gen)
        visual_sigma = self.embed2visual_sigma(to_gen).exp()

        return (audio_mu, audio_sigma), (visual_mu, visual_sigma)

class SentimentModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(SentimentModel, self).__init__()

        self.hidden1 = nn.Linear(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        x = F.relu(self.hidden1(inputs))
        x = F.tanh(self.out(x)) * 3
        # sentiment is [-3, 3] range
        return x.squeeze()

EMBEDDING_DIM = 300
AUDIO_DIM = 74
VISUAL_DIM = 43

gen_model = AudioVisualGenerator(EMBEDDING_DIM, AUDIO_DIM, VISUAL_DIM, frozen_weights=False).to(device)
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
def get_log_prob_matrix(embedding, audio, visual, data, a=1e-3):
    # assume samples in sequences are i.i.d.
    # calculate probabilities for audio and visual as if
    # sampling from distribution
    (audio_mu, audio_sigma) = audio
    (visual_mu, visual_sigma) = visual

    # exponentiate the sigma

    # calculate probabilities for text using Arora model
    # given word weights p(w), a / (a + p(w)) * v(w) . embedding

    # use softmax instead 
    # log (alpha * p(w) + (1 - alpha) exp(dotprod) / Z)
    # calc partition value Z - sum of exps of inner products of embedding with all words. Slow!

    Z_s = embedding.matmul(word_embeddings.transpose(0, 1)).exp().sum(-1)
    alpha = 1. / (Z_s * a + 1.)

    word_log_prob = 0.
    for i, sample in enumerate(data['text']):
        # sent embedding is 1D, word_weights is 1D, word_embeds is 2D
        sent_embedding = embedding[i]
        word_weights = weights[sample]
        word_embeds = word_embeddings[sample]

        # word_log_prob += (a / (a + word_weights) * word_embeds.matmul(sent_embedding)).sum()

        #for j in len(sample):
        #    word_log_prob += torch.log(alpha[i] * word_weights[j] + (1 - alpha[i]) * torch.exp(word_embeds[j].dot(sent_embedding)) / Z_s[i])

        word_log_prob += torch.log(alpha[i] * word_weights + (1 - alpha[i]) * word_embeds.matmul(sent_embedding).exp() / Z_s[i]).sum()
    print(word_log_prob)

    audio_log_prob = 0.
    if (audio_sigma == 0).sum() > 0:
        print(audio_sigma)
    for i, sample in enumerate(data['covarep']):
        audio_dist = MultivariateNormal(audio_mu[i], torch.diag(audio_sigma[i]))
        audio_log_prob += audio_dist.log_prob(sample).sum()

    visual_log_prob = 0.
    for i, sample in enumerate(data['facet']):
        visual_dist = MultivariateNormal(visual_mu[i], torch.diag(visual_sigma[i]))
        visual_log_prob += visual_dist.log_prob(sample).sum()

    total_log_prob = -(audio_log_prob + visual_log_prob + word_log_prob)
    return total_log_prob

# # eval over valid before starting
# print("Evaluating before training...")
# with torch.no_grad():
#     initial_embedding = torch.tensor(valid_embedding.copy(), device=device, dtype=torch.float32)
#     gen_model.init_embedding(initial_embedding)
#     audio, visual = gen_model()
# 
#     log_prob = get_log_probability(initial_embedding, audio, visual, valid)
#     print(log_prob)

print("Training...")
N_EPOCHS = 10

curr_embedding = torch.tensor(train_embedding.copy(), device=device, dtype=torch.float32)
gen_model.init_embedding(curr_embedding)
optimizer = optim.SGD([gen_model.embedding], lr=0.0001, momentum=0.9)
for i in range(N_EPOCHS):
    start_time = time.time()
    gen_model.zero_grad()
    epoch_loss = 0.
    for j, text, aud, vis in dataloader:
        audio, visual = gen_model(j)
        log_prob = get_log_prob_matrix(curr_embedding[j], audio, visual,
                {"text": text, "covarep": aud, "facet": vis})
        print(log_prob)
        log_prob.backward()
        optimizer.step()
        epoch_loss += log_prob
    print("epoch {}: {} ({}s)".format(i, epoch_loss, time.time() - start_time))

# print("Evaluating after training...")
# with torch.no_grad():
#     initial_embedding = torch.tensor(valid_embedding.copy(), device=device, dtype=torch.float32)
#     gen_model.init_embedding(initial_embedding)
#     audio, visual = gen_model()
# 
#     log_prob = get_log_probability(initial_embedding, audio, visual, valid)
#     print(log_prob)
# 
# print("Evaluating on test set...")
# torchify(test)
# with torch.no_grad():
#     initial_embedding = torch.tensor(test_embedding.copy(), device=device, dtype=torch.float32)
#     gen_model.init_embedding(initial_embedding)
#     audio, visual = gen_model()
# 
#     log_prob = get_log_probability(initial_embedding, audio, visual, test)
#     print(log_prob)

# sentiment analysis
senti_model = SentimentModel(EMBEDDING_DIM, 100).to(device)
senti_optimizer = optim.SGD(senti_model.parameters(), lr=0.01, momentum=0.9)
loss_function = nn.L1Loss()

print("Initial sentiment predictions")
with torch.no_grad():
    for j, senti in senti_dataloader:
        senti_predict = senti_model(curr_embedding[j])
        loss = loss_function(senti_predict, senti)

print("Training sentiment model on learned embeddings...")
N_EPOCHS = 20

for i in range(N_EPOCHS):
    senti_model.zero_grad()
    for j, senti in senti_dataloader:
        senti_predict = senti_model(curr_embedding[j])
        loss = loss_function(senti_predict, senti)
        loss.backward()
        senti_optimizer.step()

print("Sentiment predictions after training")
with torch.no_grad():
    for j, senti in senti_dataloader:
        senti_predict = senti_model(curr_embedding[j])
        loss = loss_function(senti_predict, senti)

