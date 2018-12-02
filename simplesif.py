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
from losses import get_log_prob_matrix_trimodal, get_word_log_prob_angular2
from sentiment_model import train_sentiment, train_sentiment_for_latents
from models import AudioVisualGeneratorConcat, AudioVisualGenerator, AudioVisualGeneratorMultimodal
from analyze_embeddings import get_closest_words
from utils import load_data, normalize_data, MMData, MMDataExtra, add_positional_embeddings

from sif import load_weights, get_sentence_embeddings  

def update_masks(mask_dict, data, embedding_dim):
    tmp = (data != 0).astype(int)
    mask_dict['text'] = np.broadcast_to(np.expand_dims(tmp, -1), tmp.shape + (embedding_dim,))

    print(np.all(mask_dict['text'][:,:,1] == mask_dict['text'][:,:,0]))

def update_masks_vect(mask_dict, data, key='text'):
    tmp = data != 0
    tmp2 = np.all(tmp, axis=-1).astype(int)
    print(tmp2.shape)

    mask_dict[key] = np.broadcast_to(np.expand_dims(tmp2, -1), data.shape)


def optimize_mosi():
    pass

def optimize_latents(args, train: bool, gen_model, embed_arr, dataloader, n_epochs, lr, word_prob_fn,
        device, validation_data=None, verbose=True):
    embeddings = torch.tensor(embed_arr.copy(), device=device, dtype=torch.float32)
    embeddings.requires_grad = True

    grad_params = [embeddings]
    if train and not args['freeze_weights']:
        grad_params.extend(gen_model.parameters())

    optimizer = optim.SGD(grad_params, lr=lr)

    valid_niter = 10
    start_time = time.time()
    losses = []
    for i in range(n_epochs):
        epoch_loss = 0.
        iters = 0

        # normalize embeddings after every epoch? for dot_prod loss
        for x in dataloader:
            if args['dataset'] == 'mosi':
                j, text, aud, vis, text_m, aud_m, vis_m, text_w = x
            else:
                j, text, aud, vis, text_m, aud_m, vis_m, text_w, text_a, text_a_m = x

            iters += 1
            optimizer.zero_grad()
            out = gen_model(embeddings[j])

            for modality, d in out.items():
                if d['sigma'].min().abs() < 1e-7:
                    print(d, "boo!")

            if args['dataset'] == 'mosi':
                text_gauss = text
                text_gauss_m = text_m
            else:
                text_gauss = text_a
                text_gauss_m = text_a_m

            batch_data = {
                'text': text,
                'audio': aud,
                'visual': vis,
                'text_weights': text_w,
                'audiovisual': torch.cat([aud, vis], dim=-1),
                'textaudio': torch.cat([text_gauss, aud], dim=-1),
                'textvisual': torch.cat([text_gauss, vis], dim=-1),
                'textaudiovisual': torch.cat([text_gauss, aud, vis], dim=-1),
            }

            batch_masks = {
                'text': text_m,
                'audio': aud_m,
                'visual': vis_m,
                'audiovisual': torch.cat([aud_m, vis_m], dim=-1),
                'textaudio': torch.cat([text_gauss_m, aud_m], dim=-1),
                'textvisual': torch.cat([text_gauss_m, vis_m], dim=-1),
                'textaudiovisual': torch.cat([text_gauss_m, aud_m, vis_m], dim=-1),
            }

            # TODO: check that mask and data are same size

            log_prob = -get_log_prob_matrix_trimodal(args, embeddings[j], out,
                    batch_data, batch_masks, word_prob_fn,
                    device=device, verbose=False)

            # log_prob = -get_log_prob_matrix(args, curr_embedding[j], audio, visual,
            #         {"text": text, "covarep": aud, "facet": vis}, 
            #         {"text": text_m, "covarep": aud_m, "facet": vis_m},
            #         get_word_log_prob,
            #         device=device, verbose=False)

            avg_log_prob = log_prob.mean()
            avg_log_prob.backward()

            # nn.utils.clip_grad_norm_([gen_model.embedding], 500)

            optimizer.step()
            epoch_loss += float(avg_log_prob)

        losses.append(epoch_loss)
        if i % valid_niter == 0:
            if verbose:
                print("epoch {}: {} ({}s)".format(i, epoch_loss / iters, time.time() - start_time))

            if validation_data is not None and i % (valid_niter * 8) == 0:
                valid_embedding, valid_dataloader = validation_data
                _, valid_losses = optimize_latents(args, False, gen_model, valid_embedding,
                        valid_dataloader, n_epochs, lr, word_prob_fn, device, verbose=False)
                print("Mean validation loss:", np.mean(valid_losses))

    if verbose:
        print("epoch {}: {} ({}s)".format(i, epoch_loss / iters, time.time() - start_time))
    embeddings.requires_grad = False
    return embeddings, losses



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
    parser.add_argument('dataset', choices=['mosi', 'pom', 'iemocap'])
    parser.add_argument('--pos_embed_dim', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--semi_sup_idxes', choices=['{:.1f}'.format(x) for x in np.arange(0.1, 1, 0.1)])
    parser.add_argument('--config_name', help='override config name in config file')
    parser.add_argument('--cuda_device', type=int, choices=list(range(4)))
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--sentiment_epochs', type=int)
    parser.add_argument('--emotion', choices=['happy', 'angry', 'neutral', 'sad'], help='iemocap emotion')

    args = vars(parser.parse_args())
    config = read_config(args['config_file'])
    print('######################################')
    print("Config: {}".format(config['config_num']))
    args.update(config)

    if args['sentiment_epochs']:
        args['n_sentiment_epochs'] = args['sentiment_epochs']

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
    word2ix, word_embeddings, data = load_data(args)
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
    if args['dataset'] == 'mosi':
        update_masks(train_mask, train['text'], word_embeddings.shape[-1])
        update_masks(valid_mask, valid['text'], word_embeddings.shape[-1])
        update_masks(test_mask, test['text'], word_embeddings.shape[-1])
    else:
        update_masks(train_mask, train['text_id'], word_embeddings.shape[-1])
        update_masks(valid_mask, valid['text_id'], word_embeddings.shape[-1])
        update_masks(test_mask, test['text_id'], word_embeddings.shape[-1])

    n_train = train['label'].shape[0]
    n_valid = valid['label'].shape[0]
    n_test = test['label'].shape[0]

    weights = load_weights(args)
    weights = torch.tensor(weights, device=device, dtype=torch.float32)
    word_embeddings = torch.tensor(word_embeddings, device=device, dtype=torch.float32)

    if args['word_sim_metric'] == 'dot_prod':
        word_embeddings = F.normalize(word_embeddings)

    # get sentence embeddings
    if args['dataset'] == 'mosi':
        train_embedding = get_sentence_embeddings(word_embeddings, weights, train['text'])
        valid_embedding = get_sentence_embeddings(word_embeddings, weights, valid['text'])
        test_embedding = get_sentence_embeddings(word_embeddings, weights, test['text'])

        # add random noise, since we are adding positional embeddings
        # noise = np.random.randn(train_embedding.shape[0], args['pos_embed_dim'])
        # train_embedding = np.concatenate([train_embedding, noise], axis=-1)
        # noise = np.random.randn(valid_embedding.shape[0], args['pos_embed_dim'])
        # valid_embedding = np.concatenate([valid_embedding, noise], axis=-1)
        # noise = np.random.randn(test_embedding.shape[0], args['pos_embed_dim'])
        # test_embedding = np.concatenate([test_embedding, noise], axis=-1)
    else:
        train_embedding = get_sentence_embeddings(word_embeddings, weights, train['text_id'])
        valid_embedding = get_sentence_embeddings(word_embeddings, weights, valid['text_id'])
        test_embedding = get_sentence_embeddings(word_embeddings, weights, test['text_id'])

    combined_embedding = np.concatenate([train_embedding, valid_embedding, test_embedding], axis=0)

    # we don't need the id's any more, so convert the ids into the corresponding embeddings
    if args['dataset'] == 'mosi':
        train['text_id'] = train['text']
        train['text'] = word_embeddings[train['text_id']]
        train['text_weights'] = weights[train['text_id']]

        valid['text_id'] = valid['text']
        valid['text'] = word_embeddings[valid['text_id']]
        valid['text_weights'] = weights[valid['text_id']]

        test['text_id'] = test['text']
        test['text'] = word_embeddings[test['text_id']]
        test['text_weights'] = weights[test['text_id']]
    else:
        train['text_align'] = train['text']
        train['text'] = word_embeddings[train['text_id']]
        train['text_weights'] = weights[train['text_id']]
        valid['text_align'] = valid['text']
        valid['text'] = word_embeddings[valid['text_id']]
        valid['text_weights'] = weights[valid['text_id']]
        test['text_align'] = test['text']
        test['text'] = word_embeddings[test['text_id']]
        test['text_weights'] = weights[test['text_id']]

        update_masks_vect(train_mask, train['text_align'], 'text_align')
        update_masks_vect(valid_mask, valid['text_align'], 'text_align')
        update_masks_vect(test_mask, test['text_align'], 'text_align')

    # for k, v in train.items():
    #     print(k, v.shape)
    # print('--')
    # for k, v in train_mask.items():
    #     print(k, v.shape, v.dtype)
    # print('--')

    # add positional embeddings and update masks
    if args['dataset'] == 'mosi':
        #train['text'] = add_positional_embeddings(args, train['text'])
        train['covarep'] = add_positional_embeddings(args, train['covarep'])
        train['facet'] = add_positional_embeddings(args, train['facet'])

        #valid['text'] = add_positional_embeddings(args, valid['text'])
        valid['covarep'] = add_positional_embeddings(args, valid['covarep'])
        valid['facet'] = add_positional_embeddings(args, valid['facet'])

        #test['text'] = add_positional_embeddings(args, test['text'])
        test['covarep'] = add_positional_embeddings(args, test['covarep'])
        test['facet'] = add_positional_embeddings(args, test['facet'])
        
        def update_mosi_masks(mask_dict):
            n_points, seq_len = mask_dict['covarep'].shape[:2]
            mask_extend = np.ones((n_points, seq_len, args['pos_embed_dim']), dtype=np.int64)

            mask_dict['covarep'] = np.concatenate([mask_dict['covarep'], mask_extend], axis=-1)
            mask_dict['facet'] = np.concatenate([mask_dict['facet'], mask_extend], axis=-1)
            #mask_dict['text'] = np.concatenate([mask_dict['text'], mask_extend], axis=-1)

        update_mosi_masks(train_mask)
        update_mosi_masks(valid_mask)
        update_mosi_masks(test_mask)
    else:
        train['covarep'] = add_positional_embeddings(args, train['covarep'])
        train['facet'] = add_positional_embeddings(args, train['facet'])

        valid['covarep'] = add_positional_embeddings(args, valid['covarep'])
        valid['facet'] = add_positional_embeddings(args, valid['facet'])

        test['covarep'] = add_positional_embeddings(args, test['covarep'])
        test['facet'] = add_positional_embeddings(args, test['facet'])
        
        def update_mosi_masks(mask_dict):
            n_points, seq_len = mask_dict['covarep'].shape[:2]
            mask_extend = np.ones((n_points, seq_len, args['pos_embed_dim']), dtype=np.int64)

            mask_dict['covarep'] = np.concatenate([mask_dict['covarep'], mask_extend], axis=-1)
            mask_dict['facet'] = np.concatenate([mask_dict['facet'], mask_extend], axis=-1)

        update_mosi_masks(train_mask)
        update_mosi_masks(valid_mask)
        update_mosi_masks(test_mask)

    # print()
    # for k, v in train.items():
    #     print(k, v.shape)
    # print('--')
    # for k, v in train_mask.items():
    #     print(k, v.shape, v.dtype)
    # print('--')

    """
    MOSI:
        text is already aligned
        text_id, text_weights correspond to text

    IEMOCAP/POM:
        text is the unaligned embeddings
        text_id, text_weights correspond to text
        text_align are the aligned embeddings
    """

    # combined_text = np.concatenate([train['text'], valid['text'], test['text']], axis=0)
    # combined_text_weights = np.concatenate([train['text_weights'], valid['text_weights'], test['text_weights']], axis=0)
    # combined_covarep = np.concatenate([train['covarep'], valid['covarep'], test['covarep']], axis=0)
    # combined_facet = np.concatenate([train['facet'], valid['facet'], test['facet']], axis=0)

    # print(combined_text_weights)
    # print(combined_text_weights.shape)

    # combined_masks = {
    #     'text': np.concatenate([train_mask['text'], valid_mask['text'], test_mask['text']]),
    #     'covarep': np.concatenate([train_mask['covarep'], valid_mask['covarep'], test_mask['covarep']]),
    #     'facet': np.concatenate([train_mask['facet'], valid_mask['facet'], test_mask['facet']]),
    # }

    # print closest words before training
    if word2ix is not None:
        pre_closest = get_closest_words(combined_embedding[:, :300], word_embeddings.cpu().numpy(), word2ix)

    BATCH_SIZE = args['batch_size']
    # dataset = MMData(combined_text, combined_covarep, combined_facet, combined_masks, combined_text_weights, device)
    if args['dataset'] == 'mosi':
        train_dataset = MMData(train['text'], train['covarep'], train['facet'], train_mask,
                train['text_weights'], device)
        valid_dataset = MMData(valid['text'], valid['covarep'], valid['facet'], valid_mask,
                valid['text_weights'], device)
        test_dataset = MMData(test['text'], test['covarep'], test['facet'], test_mask,
                test['text_weights'], device)
    else:
        train_dataset = MMDataExtra(train['text'], train['covarep'], train['facet'], train_mask,
                train['text_weights'], train['text_align'], device)
        valid_dataset = MMDataExtra(valid['text'], valid['covarep'], valid['facet'], valid_mask,
                valid['text_weights'], valid['text_align'], device)
        test_dataset = MMDataExtra(test['text'], test['covarep'], test['facet'], test_mask,
                test['text_weights'], test['text_align'], device)

    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE * 8)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 8)

    print("# batches: {}".format(len(train_dataset) // BATCH_SIZE))

    """
    2. Initialize regression model to generate mean and variance of audio and
        visual features
    """

    EMBEDDING_DIM = train['text'].shape[-1]
    AUDIO_DIM = train['covarep'].shape[-1]
    VISUAL_DIM = train['facet'].shape[-1]

    print(EMBEDDING_DIM)

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
        word_log_prob_fn = get_word_log_prob_angular2
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

    def get_word_log_prob2(latents, word_weights, sent_embeddings, mask):
        word_log_prob = word_log_prob_fn(latents, word_embeddings, word_weights, sent_embeddings, mask, a)
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

            #curr_embedding = torch.tensor(combined_embedding.copy(), device=device, dtype=torch.float32)
            # curr_embedding = torch.tensor(train_embedding.copy(), device=device, dtype=torch.float32)

            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # print("Initial sentiment predictions, before optimizing audio and visual")
            # train_sentiment_for_latents(args, curr_embedding, sentiment_data, device,
            #         (n_train, n_valid, n_test), train_idxes=sentiment_train_idxes,
            #         model_save_path=pre_path)

            # save initial embeddings
            torch.save(torch.tensor(combined_embedding.copy(), device=device, dtype=torch.float32),
                    os.path.join(pre_path, 'embed.bin'))

            #gen_model = AudioVisualGenerator(EMBEDDING_DIM, AUDIO_DIM, VISUAL_DIM,
            #        frozen_weights=args['freeze_weights']).to(device)

            gen_model = AudioVisualGeneratorMultimodal(EMBEDDING_DIM, AUDIO_DIM, VISUAL_DIM,
                    frozen_weights=args['freeze_weights']).to(device)

            print("Training...")

            lr = args['lr']

            N_EPOCHS = args['n_epochs']
            N_EPOCHS = 10

            train_embed, train_losses = optimize_latents(args, True, gen_model, train_embedding,
                    dataloader, N_EPOCHS, lr, get_word_log_prob2, device,
                    validation_data=(valid_embedding, valid_dataloader))

            with open(os.path.join(folder, 'embed_loss.txt'), 'w') as f:
                for loss in train_losses:
                    f.write('{}\n'.format(loss))

            if word2ix is not None:
                post_closest = get_closest_words(train_embed[:, :300].cpu().numpy(),
                        word_embeddings.cpu().numpy(), word2ix)

            with open(os.path.join(folder, 'closest_words.txt'), 'w') as f:
                for pre, post in zip(pre_closest, post_closest):
                    f.write('{}\t{}\n'.format(' '.join(pre), ' '.join(post)))

            valid_embed, _ = optimize_latents(args, False, gen_model, valid_embedding,
                    valid_dataloader, N_EPOCHS, lr, get_word_log_prob2, device)
            test_embed, test_losses = optimize_latents(args, False, gen_model, test_embedding,
                    test_dataloader, N_EPOCHS, lr, get_word_log_prob2, device)

            torch.save(torch.cat([train_embed, valid_embed, test_embed], dim=0),
                    os.path.join(post_path, 'embed.bin'))

            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("Initial sentiment predictions, AFTER optimizing audio and visual")
            latents = (train_embed, valid_embed, test_embed)
            train_sentiment_for_latents(args, latents, sentiment_data, device,
                    train_idxes=sentiment_train_idxes,
                    model_save_path=post_path)
    else:
        raise NotImplementedError

#         # sentiment analysis
#         AUDIO_EMBEDDING_DIM = 20
#         VISUAL_EMBEDDING_DIM = 20
# 
#         gen_model = AudioVisualGeneratorConcat(AUDIO_EMBEDDING_DIM, VISUAL_EMBEDDING_DIM, AUDIO_DIM,
#                         VISUAL_DIM, frozen_weights=True).to(device)
#         text_embedding = torch.tensor(train_embedding.copy(), device=device, dtype=torch.float32)
# 
#         curr_embedding = gen_model.init_embeddings(text_embedding)
# 
#         print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
#         print("Initial sentiment predictions, before optimizing audio and visual")
#         train_sentiment_for_latents(args, curr_embedding, senti_dataloader, device)
# 
#         #gen_model = AudioVisualGenerator(EMBEDDING_DIM, AUDIO_DIM, VISUAL_DIM, frozen_weights=True).to(device)
# 
#         print("Training...")
# 
#         print(curr_embedding[:5, :5])
#         print(curr_embedding[:5, 305:310])
# 
#         curr_embedding.requires_grad = True
# 
#         lr = args['lr']
#         lr = 1e-6
#         optimizer = optim.SGD([curr_embedding], lr=lr)
#         # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
# 
#         N_EPOCHS = args['n_epochs']
#         #N_EPOCHS = 200
#         start_time = time.time()
# 
#         """
#         Assumption: the word embedding is already optimized (since derived from MLE).
#         So we just optimize the audio and visual.
#         """
# 
#         for i in range(N_EPOCHS):
#             epoch_loss = 0.
#             iters = 0
#             #curr_embedding = F.normalize(curr_embedding)
#             for j, text, aud, vis in dataloader:
#                 iters += 1
#                 optimizer.zero_grad()
#                 # print(curr_embedding[:10])
# 
#                 audio, visual = gen_model(curr_embedding[j, 300:320], curr_embedding[j, 320:])
# 
#                 if audio[1].min().abs() < 1e-7:
#                     print("boo!")
#                 if visual[1].min().abs() < 1e-7:
#                     print("boot!")
# 
#                 # only optimize audio and visual?
#                 # if optimize word also, seems to blow up
#                 log_prob = -get_log_prob_matrix(args, curr_embedding[j, :300], audio, visual,
#                         {"text": text, "covarep": aud, "facet": vis}, word_embeddings, weights, device=device)
# 
#                 avg_log_prob = log_prob.mean()
#                 avg_log_prob.backward(retain_graph=True)
# 
#                 # nn.utils.clip_grad_norm_([gen_model.embedding], 500)
#                 # print(curr_embedding.grad[:5, :5])
# 
#                 optimizer.step()
#                 epoch_loss += avg_log_prob
#             # scheduler.step()
# 
#             if i % valid_niter == 0:
#                 print("epoch {}: {} ({}s)".format(i, epoch_loss / iters, time.time() - start_time))
#         curr_embedding.requires_grad = False
# 
#         print(curr_embedding[:5, :5])
#         print(curr_embedding[:5, 305:310])
# 
#         print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#         print("Initial sentiment predictions, AFTER optimizing audio and visual")
#         train_sentiment_for_latents(args, curr_embedding, senti_dataloader, device, verbose=True)

    sys.stdout.flush()

if __name__ == '__main__':
    main()
