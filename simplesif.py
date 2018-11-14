import os
import json
import time
import sys
import argparse
import pprint

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

from losses import get_log_prob_matrix, get_word_log_prob_angular, get_word_log_prob_dot_prod
from sentiment_model import train_sentiment, eval_sentiment, train_sentiment_for_latents
from models import AudioVisualGeneratorConcat, AudioVisualGenerator
from analyze_embeddings import get_closest_words
from utils import load_data, normalize_data, MMData

from sif import load_weights, get_word_embeddings  

def update_masks(mask_dict, data):
    mask_dict['text'] = (data != 0).astype(int)

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
    config = json.load(open(config_file, 'r'))

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(config)

    return config

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

def main():
    args = parse_arguments()

    if args['cuda_device']:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args['cuda_device'])

    device = torch.device('cuda')

    """
    Procedure:
    1. Initialize sentence embedding using the SIF algorithm
    2. Initialize linear regression model to mean and variance (element-wise independent)
        of audio and visual features
    3. Maximize joint log-probability of generating sentence, audio, and visual features.
    """

    # Load data
    word2ix, word_embeddings, data = load_data()
    train, valid, test = data

    """
    Combine data:

    Since we are comparing performance of the sentiment model before we add in audio and visual,
    and after, we should optimize all embeddings at once.

    But for sentiment, we should still keep the same train/valid/test split.
    """

    # Normalize audio and visual features to [-1, 1], remove unused features.
    train, train_mask = normalize_data(train)
    valid, valid_mask = normalize_data(valid)
    test, test_mask = normalize_data(test)

    # get mask for text data
    update_masks(train_mask, train['text'])
    update_masks(valid_mask, valid['text'])
    update_masks(test_mask, test['text'])

    n_train = train['label'].shape[0]
    n_valid = valid['label'].shape[0]
    n_test = test['label'].shape[0]

    combined_text = np.concatenate([train['text'], valid['text'], test['text']], axis=0)
    combined_covarep = np.concatenate([train['covarep'], valid['covarep'], test['covarep']], axis=0)
    combined_facet = np.concatenate([train['facet'], valid['facet'], test['facet']], axis=0)

    combined_masks = {
        'text': np.concatenate([train_mask['text'], valid_mask['text'], test_mask['text']]),
        'covarep': np.concatenate([train_mask['covarep'], valid_mask['covarep'], test_mask['covarep']]),
        'facet': np.concatenate([train_mask['facet'], valid_mask['facet'], test_mask['facet']]),
    }

    weights = load_weights()
    weights = torch.tensor(weights, device=device, dtype=torch.float32)
    word_embeddings = torch.tensor(word_embeddings, device=device, dtype=torch.float32)

    if args['word_sim_metric'] == 'dot_prod':
        word_embeddings = F.normalize(word_embeddings)

    # get sentence embeddings
    train_embedding = get_word_embeddings(word_embeddings, weights, train['text'])
    valid_embedding = get_word_embeddings(word_embeddings, weights, valid['text'])
    test_embedding = get_word_embeddings(word_embeddings, weights, test['text'])
    combined_embedding = np.concatenate([train_embedding, valid_embedding, test_embedding], axis=0)

    # print closest words before training
    pre_closest = get_closest_words(combined_embedding, word_embeddings.cpu().numpy(), word2ix)

    BATCH_SIZE = args['batch_size']
    # dataset = MMData(train['text'], train['covarep'], train['facet'], device)
    dataset = MMData(combined_text, combined_covarep, combined_facet, combined_masks, device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("# batches: {}".format(len(dataset) // BATCH_SIZE))

    """
    2. Initialize regression model to generate mean and variance of audio and
        visual features
    """

    EMBEDDING_DIM = word_embeddings.size()[-1]
    AUDIO_DIM = train['covarep'].shape[-1]
    VISUAL_DIM = train['facet'].shape[-1]

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

    # initialize probability function for word embeddings
    if args['word_sim_metric'] == 'angular':
        word_log_prob_fn = get_word_log_prob_angular
    elif args['word_sim_metric'] == 'dot_prod':
        word_log_prob_fn = get_word_log_prob_dot_prod
    else:
        raise NotImplementedError

    a = 1e-3

    def get_word_log_prob(latents, text, mask):
        word_log_prob = word_log_prob_fn(latents, weights, word_embeddings, text, mask, a)
        if word_log_prob.min().abs() == np.inf:
            print('word inf')
            print(latents.size())
            print(latents.matmul(word_embeddings.transpose(0, 1)).max())
            print(latents.matmul(word_embeddings.transpose(0, 1)).exp().max())
            print(latents)
            sys.exit()

        return word_log_prob

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

            json.dump(args, open(os.path.join(folder, 'config.json'), 'w'), indent=2)
            
            pre_path = os.path.join(folder, 'pre')
            post_path = os.path.join(folder, 'post')

            if not os.path.isdir(pre_path):
                os.mkdir(pre_path)
            if not os.path.isdir(post_path):
                os.mkdir(post_path)

            curr_embedding = torch.tensor(combined_embedding.copy(), device=device, dtype=torch.float32)

            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("Initial sentiment predictions, before optimizing audio and visual")
            train_sentiment_for_latents(args, curr_embedding, sentiment_data, device,
                    (n_train, n_valid, n_test), train_idxes=sentiment_train_idxes,
                    model_save_path=pre_path)

            # save initial embeddings
            torch.save(curr_embedding, os.path.join(pre_path, 'embed.bin'))

            gen_model = AudioVisualGenerator(EMBEDDING_DIM, AUDIO_DIM, VISUAL_DIM,
                    frozen_weights=args['freeze_weights']).to(device)

            print("Training...")

            curr_embedding.requires_grad = True

            lr = args['lr']

            grad_params = [curr_embedding]
            if not args['freeze_weights']:
                grad_params.extend(gen_model.parameters())
            optimizer = optim.SGD(grad_params, lr=lr)
            # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)

            N_EPOCHS = args['n_epochs']
            start_time = time.time()
            train_losses = []
            for i in range(N_EPOCHS):
                epoch_loss = 0.
                iters = 0
                #curr_embedding = F.normalize(curr_embedding)
                for j, text, aud, vis, text_m, aud_m, vis_m in dataloader:
                    iters += 1
                    optimizer.zero_grad()

                    audio, visual = gen_model(curr_embedding[j])

                    if audio[1].min().abs() < 1e-7:
                        print("boo!")
                    if visual[1].min().abs() < 1e-7:
                        print("boot!")

                    log_prob = -get_log_prob_matrix(args, curr_embedding[j], audio, visual,
                            {"text": text, "covarep": aud, "facet": vis}, 
                            {"text": text_m, "covarep": aud_m, "facet": vis_m},
                            get_word_log_prob,
                            device=device, verbose=False)

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

if __name__ == '__main__':
    main()
