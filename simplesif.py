import os
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

import data_loader as loader

sys.path.append('SIF/src')
import SIF_embedding, params, data_io

torch.cuda.set_device(3)
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

# if os.path.isfile('train.pkl'):
#     train = pickle.load(open('train.pkl', 'rb'))
#     valid = pickle.load(open('valid.pkl', 'rb'))
#     test = pickle.load(open('test.pkl', 'rb'))
# else:
#     pickle.dump(train, 

# normalize audio and visual features to [-1, 1]
audio_min = train['covarep'].min((0, 1))
audio_max = train['covarep'].max((0, 1))
print(audio_max - audio_min)
audio_diff = audio_max - audio_min
audio_nonzeros = (audio_diff == 0).nonzero()[0]
print(audio_nonzeros)
audio_nonzeros = audio_diff.nonzero()[0]
print(train['covarep'].shape)
print(train['covarep'][:, :, audio_nonzeros].shape)

train['covarep'] = train['covarep'][:, :, audio_nonzeros]

audio_min = train['covarep'].min((0, 1))
audio_max = train['covarep'].max((0, 1))
audio_diff = audio_max - audio_min

vis_min = train['facet'].min((0, 1))
vis_max = train['facet'].max((0, 1))

# print(train['covarep'].shape)
# train['covarep'][:, :, audio_nonzeros] = (train['covarep'][:, :, audio_nonzeros] + audio_min[audio_nonzeros]) * 2. / audio_diff[audio_nonzeros] - 1.
# print(train['covarep'].shape)
train['covarep'] = (train['covarep'] + audio_min) * 2. / (audio_max - audio_min) - 1.
train['facet'] = (train['facet'] + vis_min) * 2. / (vis_max - vis_min) - 1.


def get_prefix(d, n):
    return {k: v[:n] for k, v in d.items()}

#train = get_prefix(train, 32)

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

BATCH_SIZE = 32
dataset = MMData(train['text'], train['covarep'], train['facet'], device)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
senti_dataset = SentimentData(train['label'], device)
senti_dataloader = DataLoader(senti_dataset, batch_size=BATCH_SIZE, shuffle=True)

epsilon = torch.tensor(1e-6, device=device)

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
AUDIO_DIM = train['covarep'].shape[-1]
VISUAL_DIM = train['facet'].shape[-1]
print(AUDIO_DIM)
print(VISUAL_DIM)

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
def get_log_prob_matrix(latents, audio, visual, data, a=1e-3):
    """
    Return the log probability for the batch data given the
    latent variables, and the derived audio and visual parameters.

    Returns one log-probability value per example in the batch.

    Arguments:
        latents: the (joint) latent variables for the batch
        audio: the audio params for the batch (tuple mu, sigma)
        visual: the visual params for the batch (tuple mu, sigma)
        data: dict containing text, audio and visual features.
            text is a tensor of word ids, covarep is a tensor of audio
            features, facet is a tensor of visual features.
    """
    (audio_mu, audio_sigma) = audio
    (visual_mu, visual_sigma) = visual
    audio_sigma = audio_sigma + epsilon
    visual_sigma = visual_sigma + epsilon

    # exponentiate the sigma

    # calculate probabilities for text using Arora model
    # given word weights p(w), a / (a + p(w)) * v(w) . embedding

    # use softmax instead 
    # log (alpha * p(w) + (1 - alpha) exp(dotprod) / Z)
    # calc partition value Z - sum of exps of inner products of embedding with all words. Slow!

    Z_s = latents.matmul(word_embeddings.transpose(0, 1)).exp().sum(-1, keepdim=True)
    alpha = 1 / (Z_s * a + 1)

    word_log_prob = 0

    word_weights = weights[data['text']]
    sent_embeddings = word_embeddings[data['text']]

    unigram_prob = alpha * word_weights

    dot_prod = torch.bmm(sent_embeddings, latents.unsqueeze(-1)).squeeze()
    context_prob = (1 - alpha) * dot_prod.exp() / Z_s

    log_probs = torch.log(unigram_prob + context_prob)
    word_log_prob = log_probs.sum(dim=-1)

    # assume samples in sequences are i.i.d.
    # calculate probabilities for audio and visual as if
    # sampling from distribution

    # audio: (batch, seqlength, n_features)
    # audio_mu: (batch, n_features)
    # audio_sigma: (batch, n_features)
    # independent normals, so just calculate log prob directly

    # audio log prob
    sig_sq = audio_sigma.pow(2).unsqueeze(1)
    term1 = torch.log(1 / torch.sqrt(2 * np.pi * sig_sq)) 
    
    diff = data['covarep'] - audio_mu.unsqueeze(1)
    term2 = diff.pow(2) / (2 * sig_sq)

    audio_log_prob = term1 - term2
    audio_log_prob = audio_log_prob.squeeze().sum(-1).sum(-1)

    # visual log prob
    vis_sig_sq = visual_sigma.pow(2).unsqueeze(1)
    term1 = torch.log(1 / torch.sqrt(2 * np.pi * vis_sig_sq))

    diff = data['facet'] - visual_mu.unsqueeze(1)
    term2 = diff.pow(2) / (2 * vis_sig_sq)

    visual_log_prob = (term1 - term2).squeeze().sum(-1).sum(-1)

    # final output: one value per datapoint
    total_log_prob = audio_log_prob + visual_log_prob + word_log_prob
    return total_log_prob


gen_model = AudioVisualGenerator(EMBEDDING_DIM, AUDIO_DIM, VISUAL_DIM, frozen_weights=True).to(device)
#gen_model = nn.DataParallel(gen_model)

print("Training...")
N_EPOCHS = 100

curr_embedding = torch.tensor(train_embedding.copy(), device=device, dtype=torch.float32)
print("Initial word embeddings:", curr_embedding.size())

gen_model.init_embedding(curr_embedding)
optimizer = optim.SGD([gen_model.embedding], lr=1e-3)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)

for i in range(N_EPOCHS):
    start_time = time.time()
    epoch_loss = 0.
    iters = 0
    for j, text, aud, vis in dataloader:
        iters += 1
        gen_model.zero_grad()
        # print(curr_embedding[:10])
        audio, visual = gen_model(j)

        if audio[1].min().abs() < 1e-7:
            print("boo!")
        if visual[1].min().abs() < 1e-7:
            print("boot!")

        log_prob = -get_log_prob_matrix(curr_embedding[j], audio, visual,
                {"text": text, "covarep": aud, "facet": vis})

        #print(log_prob.max())

        avg_log_prob = log_prob.mean()
        avg_log_prob.backward()

        # log_prob.backward()
        optimizer.step()
        epoch_loss += avg_log_prob
    # scheduler.step()
    print("epoch {}: {} ({}s)".format(i, epoch_loss / iters, time.time() - start_time))

# sentiment analysis
senti_model = SentimentModel(EMBEDDING_DIM, 100).to(device)
senti_optimizer = optim.SGD(senti_model.parameters(), lr=1e-4)
loss_function = nn.L1Loss()

print("Initial sentiment predictions")
total_loss = 0
with torch.no_grad():
    for j, senti in senti_dataloader:
        senti_predict = senti_model(curr_embedding[j])
        loss = loss_function(senti_predict, senti)
        total_loss += loss
print(total_loss)

print("Training sentiment model on learned embeddings...")
N_EPOCHS = 20

for i in range(N_EPOCHS):
    epoch_loss = 0
    for j, senti in senti_dataloader:
        senti_model.zero_grad()
        senti_predict = senti_model(curr_embedding[j])
        loss = loss_function(senti_predict, senti)
        epoch_loss += loss
        loss.backward()
        senti_optimizer.step()
    print("Epoch {}: {}".format(i, epoch_loss))

print("Sentiment predictions after training")
total_loss = 0
with torch.no_grad():
    for j, senti in senti_dataloader:
        senti_predict = senti_model(curr_embedding[j])
        loss = loss_function(senti_predict, senti)
        total_loss += loss
print(total_loss)
