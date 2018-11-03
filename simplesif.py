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
import h5py

import data_loader as loader
from losses import get_log_prob_matrix
from sentiment_model import train_sentiment, eval_sentiment, train_sentiment_for_latents
from models import AudioVisualGeneratorConcat
from analyze_embeddings import get_closest_words

from sif import load_weights, get_word_embeddings  

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
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--semi_sup_idxes', choices=['{:.1f}'.format(x) for x in np.arange(0.1, 1, 0.1)])
    parser.add_argument('--config_name', help='override config name in config file')
    parser.add_argument('--cuda_device', type=int, choices=list(range(4)))
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--early_stopping', action='store_true')

    args = vars(parser.parse_args())
    config = read_config(args['config_file'])
    print('######################################')
    print("Config: {}".format(config['config_num']))
    args.update(config)

    return args

args = parse_arguments()

if args['cuda_device']:
    torch.cuda.set_device(args['cuda_device'])
else:
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

Note: loader.load_word_level_features always returns the same split.
"""
MAX_SEGMENT_LEN = args['seq_len']
TR_PROPORTION = 2.0/3

def load_data():
    word2ix = loader.load_word2ix()

    # load glove 300d word embeddings
    word_embeddings = loader.load_word_embedding()
    
    # TODO: save train, valid, test in file so that don't have to
    # regenerate each time
    #train, valid, test = loader.load_word_level_features(max_len, tr_proportion)

    with h5py.File('mosi_data.h5', 'r') as f:
        keys = [
            'facet',
            'covarep',
            'text',
            'lengths',
            'label',
            'id',
        ]
        train = {}
        valid = {}
        test = {}

        for k in keys:
            train[k] = f['train'][k][:]
            valid[k] = f['valid'][k][:]
            test[k] = f['test'][k][:]

    return word2ix, word_embeddings, (train, valid, test)

word2ix, word_embeddings, data = load_data()
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

n_train = train['label'].shape[0]
n_valid = valid['label'].shape[0]
n_test = test['label'].shape[0]

combined_text = np.concatenate([train['text'], valid['text'], test['text']], axis=0)
combined_covarep = np.concatenate([train['covarep'], valid['covarep'], test['covarep']], axis=0)
combined_facet = np.concatenate([train['facet'], valid['facet'], test['facet']], axis=0)

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

weights = load_weights()
weights = torch.tensor(weights, device=device, dtype=torch.float32)
word_embeddings = torch.tensor(word_embeddings, device=device, dtype=torch.float32)

if args['word_sim_metric'] == 'dot_prod':
    word_embeddings = F.normalize(word_embeddings)

train_embedding = get_word_embeddings(word_embeddings, weights, train['text'])
valid_embedding = get_word_embeddings(word_embeddings, weights, valid['text'])
test_embedding = get_word_embeddings(word_embeddings, weights, test['text'])
combined_embedding = np.concatenate([train_embedding, valid_embedding, test_embedding], axis=0)

BATCH_SIZE = args['batch_size']
# dataset = MMData(train['text'], train['covarep'], train['facet'], device)
dataset = MMData(combined_text, combined_covarep, combined_facet, device)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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

sentiment_data = (train['label'], valid['label'], test['label'])
sentiment_train_idxes = None
if args['semi_sup_idxes'] is not None:
    with h5py.File('subset_idxes.h5', 'r') as f:
        sentiment_train_idxes = f[args['semi_sup_idxes']][:]
        print(sentiment_train_idxes.shape)

joint = True
if joint:
    for i in range(args['n_runs']):
        if args['config_name']:
            config_name = args['config_name']
        else:
            config_name = os.path.split(os.path.split(args['config_file'])[0])[1]
        
        folder = 'model_saves/{}/config_{}_run_{}'.format(config_name, args['config_num'], i)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        json.dump(args, open(os.path.join(folder, 'config.json'), 'w'))
        
        pre_path = os.path.join(folder, 'pre')
        post_path = os.path.join(folder, 'post')

        if not os.path.isdir(pre_path):
            os.mkdir(pre_path)
        if not os.path.isdir(post_path):
            os.mkdir(post_path)

        curr_embedding = torch.tensor(combined_embedding.copy(), device=device, dtype=torch.float32)

        # print closest words before training
        pre_closest = get_closest_words(curr_embedding.cpu().numpy(), word_embeddings.cpu().numpy(), word2ix)

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("Initial sentiment predictions, before optimizing audio and visual")
        train_sentiment_for_latents(args, curr_embedding, sentiment_data, device,
                (n_train, n_valid, n_test), train_idxes=sentiment_train_idxes,
                model_save_path=pre_path)

        # save initial embeddings
        torch.save(curr_embedding, os.path.join(pre_path, 'embed.bin'))

        gen_model = AudioVisualGenerator(EMBEDDING_DIM, AUDIO_DIM, VISUAL_DIM, frozen_weights=True).to(device)

        print("Training...")

        curr_embedding.requires_grad = True

        lr = args['lr']
        optimizer = optim.SGD([curr_embedding], lr=lr)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)

        N_EPOCHS = args['n_epochs']
        start_time = time.time()
        train_losses = []
        for i in range(N_EPOCHS):
            epoch_loss = 0.
            iters = 0
            #curr_embedding = F.normalize(curr_embedding)
            for j, text, aud, vis in dataloader:
                iters += 1
                optimizer.zero_grad()

                audio, visual = gen_model(curr_embedding[j])

                if audio[1].min().abs() < 1e-7:
                    print("boo!")
                if visual[1].min().abs() < 1e-7:
                    print("boot!")

                log_prob = -get_log_prob_matrix(args, curr_embedding[j], audio, visual,
                        {"text": text, "covarep": aud, "facet": vis}, word_embeddings, weights, device=device, verbose=False)

                avg_log_prob = log_prob.mean()
                #avg_log_prob.backward(retain_graph=True)
                avg_log_prob.backward()

                # nn.utils.clip_grad_norm_([gen_model.embedding], 500)

                optimizer.step()
                epoch_loss += avg_log_prob
            # scheduler.step()
            train_losses.append(epoch_loss)
            if i % valid_niter == 0:
                print("epoch {}: {} ({}s)".format(i, epoch_loss / iters, time.time() - start_time))
        print("epoch {}: {} ({}s)".format(i, epoch_loss / iters, time.time() - start_time))
        curr_embedding.requires_grad = False

        with open(os.path.join(folder, 'embed_loss.txt'), 'w') as f:
            for loss in train_losses:
                f.write('{}\n'.format(loss))
        torch.save(curr_embedding, os.path.join(post_path, 'embed.bin'))

        post_closest = get_closest_words(curr_embedding.cpu().numpy(), word_embeddings.cpu().numpy(), word2ix)

        with open(os.path.join(folder, 'closest_words.txt'), 'w') as f:
            for pre, post in zip(pre_closest, post_closest):
                f.write('{}\t{}\n'.format(' '.join(pre), ' '.join(post)))

        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Initial sentiment predictions, AFTER optimizing audio and visual")
        train_sentiment_for_latents(args, curr_embedding, sentiment_data, device,
                (n_train, n_valid, n_test), train_idxes=sentiment_train_idxes,
                model_save_path=post_path)
else:
    raise NotImplementedError

    # sentiment analysis
    AUDIO_EMBEDDING_DIM = 20
    VISUAL_EMBEDDING_DIM = 20

    gen_model = AudioVisualGeneratorConcat(AUDIO_EMBEDDING_DIM, VISUAL_EMBEDDING_DIM, AUDIO_DIM,
                    VISUAL_DIM, frozen_weights=True).to(device)
    text_embedding = torch.tensor(train_embedding.copy(), device=device, dtype=torch.float32)

    curr_embedding = gen_model.init_embeddings(text_embedding)

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Initial sentiment predictions, before optimizing audio and visual")
    train_sentiment_for_latents(args, curr_embedding, senti_dataloader, device)

    #gen_model = AudioVisualGenerator(EMBEDDING_DIM, AUDIO_DIM, VISUAL_DIM, frozen_weights=True).to(device)

    print("Training...")

    print(curr_embedding[:5, :5])
    print(curr_embedding[:5, 305:310])

    curr_embedding.requires_grad = True

    lr = args['lr']
    lr = 1e-6
    optimizer = optim.SGD([curr_embedding], lr=lr)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)

    N_EPOCHS = args['n_epochs']
    #N_EPOCHS = 200
    start_time = time.time()

    """
    Assumption: the word embedding is already optimized (since derived from MLE).
    So we just optimize the audio and visual.
    """

    for i in range(N_EPOCHS):
        epoch_loss = 0.
        iters = 0
        #curr_embedding = F.normalize(curr_embedding)
        for j, text, aud, vis in dataloader:
            iters += 1
            optimizer.zero_grad()
            # print(curr_embedding[:10])

            audio, visual = gen_model(curr_embedding[j, 300:320], curr_embedding[j, 320:])

            if audio[1].min().abs() < 1e-7:
                print("boo!")
            if visual[1].min().abs() < 1e-7:
                print("boot!")

            # only optimize audio and visual?
            # if optimize word also, seems to blow up
            log_prob = -get_log_prob_matrix(args, curr_embedding[j, :300], audio, visual,
                    {"text": text, "covarep": aud, "facet": vis}, word_embeddings, weights, device=device)

            avg_log_prob = log_prob.mean()
            avg_log_prob.backward(retain_graph=True)

            # nn.utils.clip_grad_norm_([gen_model.embedding], 500)
            # print(curr_embedding.grad[:5, :5])

            optimizer.step()
            epoch_loss += avg_log_prob
        # scheduler.step()

        if i % valid_niter == 0:
            print("epoch {}: {} ({}s)".format(i, epoch_loss / iters, time.time() - start_time))
    curr_embedding.requires_grad = False

    print(curr_embedding[:5, :5])
    print(curr_embedding[:5, 305:310])

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("Initial sentiment predictions, AFTER optimizing audio and visual")
    train_sentiment_for_latents(args, curr_embedding, senti_dataloader, device, verbose=True)

sys.stdout.flush()

