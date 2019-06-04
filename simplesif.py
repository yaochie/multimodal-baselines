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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import numpy as np
import h5py

from losses import get_log_prob_matrix, get_word_log_prob_angular, get_word_log_prob_dot_prod
# from losses import get_log_prob_matrix_trimodal, get_word_log_prob_angular2
from losses import get_word_log_prob_angular2
from sentiment_model import train_sentiment, train_sentiment_for_latents, SentimentData
from sentiment_model import SentimentModel
from models import AudioVisualGeneratorConcat, AudioVisualGenerator, AudioVisualGeneratorMultimodal
from analyze_embeddings import get_closest_words
from utils import load_data, normalize_data, MMData, MMDataExtra, add_positional_embeddings

from sif import load_weights, get_sentence_embeddings  
from sif2 import estimate_embedding_overall, estimate_embedding_overall_gpu, estimate_embedding_overall_gpu2

def update_masks(mask_dict, data, embedding_dim):
    tmp = (data != 0).astype(int)
    mask_dict['text'] = np.broadcast_to(np.expand_dims(tmp, -1), tmp.shape + (embedding_dim,))

    print(np.all(mask_dict['text'][:,:,1] == mask_dict['text'][:,:,0]))

def update_masks_vect(mask_dict, data, key='text'):
    tmp = data != 0
    tmp2 = np.all(tmp, axis=-1).astype(int)
    print(tmp2.shape)

    mask_dict[key] = np.broadcast_to(np.expand_dims(tmp2, -1), data.shape)

def optimize_latents(args, train: bool, gen_model, embed_arr, dataloader, n_epochs, lr, word_prob_fn,
        device, validation_data=None, verbose=True):
    embeddings = torch.tensor(embed_arr.copy(), device=device, dtype=torch.float32)
    embeddings.requires_grad = True

    grad_params = [embeddings]
    if train and not args['freeze_weights']:
        grad_params.extend(gen_model.parameters())

    if args['optimizer'] == 'sgd':
        optimizer = optim.SGD(grad_params, lr=lr)
    elif args['optimizer'] == 'adam':
        optimizer = optim.Adam(grad_params, lr=lr)

    valid_niter = 10
    start_time = time.time()
    losses = []
    all_valid_losses = []
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

            if not args['unimodal']:
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

            else:
                batch_data = {
                    'text': text,
                    'audio': aud,
                    'visual': vis,
                    'text_weights': text_w,
                }

                batch_masks = {
                    'text': text_m,
                    'audio': aud_m,
                    'visual': vis_m,
                }

            log_prob = -get_log_prob_matrix(args, embeddings[j], out,
                    batch_data, batch_masks, word_prob_fn,
                    device=device, verbose=False)

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
                print("Validation loss:", valid_losses[-1])
                all_valid_losses.append(valid_losses[-1])

    # Final validation
    if validation_data is not None:
        valid_embedding, valid_dataloader = validation_data
        _, valid_losses = optimize_latents(args, False, gen_model, valid_embedding,
                valid_dataloader, n_epochs, lr, word_prob_fn, device, verbose=False)
        print("(Final) Validation loss:", valid_losses[-1])
        all_valid_losses.append(valid_losses[-1])

    embeddings.requires_grad = False
    return embeddings, (losses, all_valid_losses)



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
    parser.add_argument('config_file', help='JSON file containing hyperparameters for model')
    parser.add_argument('dataset', choices=['mosi', 'pom', 'iemocap'])
    parser.add_argument('--unimodal', action='store_true', help='run mmb1 (unimodal factorization)')
    parser.add_argument('--pos_embed_dim', type=int)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--semi_sup_idxes', choices=['{:.1f}'.format(x) for x in np.arange(0.1, 1, 0.1)])
    parser.add_argument('--config_name', help='override config name in config file')
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--early_stopping', action='store_true',
                        help='early stopping when training sentiment model')
    parser.add_argument('--sentiment_epochs', type=int)
    parser.add_argument('--emotion', choices=['happy', 'angry', 'neutral', 'sad'], help='iemocap emotion')
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('--norm', choices=['layer_norm', 'batch_norm'])
    parser.add_argument('--likelihood_weight', type=float)
    parser.add_argument('--e2e', choices=['y', 'n'], help='end-to-end training of latent variables')
    parser.add_argument('--time_test', action='store_true', help='Run inference timing')

    parser.add_argument('--cuda_device', type=int, choices=list(range(4)), help='set CUDA device number')
    parser.add_argument('--cuda', action='store_true')

    args = vars(parser.parse_args())

    override_dict = {}

    if args['pos_embed_dim'] is not None:
        override_dict['pos_embed_dim'] = args['pos_embed_dim']
    if args['e2e'] is not None:
        override_dict['e2e'] = args['e2e']

    if args['likelihood_weight'] is not None:
        like_weight = args['likelihood_weight']
    else:
        like_weight = None

    config = read_config(args['config_file'])
    print('######################################')
    print("Config: {}".format(config['config_num']))
    args.update(config)

    args.update(override_dict)
    if args['e2e'] == 'y':
        args['e2e'] = True
    elif args['e2e'] == 'n':
        args['e2e'] = False

    if args['sentiment_epochs']:
        args['n_sentiment_epochs'] = args['sentiment_epochs']

    return args

def main():
    args = parse_arguments()

    if args['cuda_device']:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args['cuda_device'])

    if args['cuda']:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

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
    if args['word_sim_metric'] == 'dot_prod':
        word_embeddings = word_embeddings / np.linalg.norm(word_embeddings, axis=-1, keepdims=True)

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

    weights = torch.tensor(weights, device=device, dtype=torch.float32)
    word_embeddings = torch.tensor(word_embeddings, device=device, dtype=torch.float32)

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
    print("# pos embeddings:", args['pos_embed_dim'])
    if 'pos_embed_dim' in args and args['pos_embed_dim'] > 0:
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
    else:
        print("not adding positional embeddings!")

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

    sentiment_data = (train['label'], valid['label'], test['label'])
    sentiment_train_idxes = None
    if args['dataset'] == 'mosi':
        senti_mask = torch.zeros(n_train, device=device)
    else:
        senti_mask = torch.zeros(n_train, 1, device=device)

    if args['semi_sup_idxes'] is not None:
        idxes_file = '{}_subset_idxes.h5'.format(args['dataset'])
        with h5py.File(idxes_file, 'r') as f:
            sentiment_train_idxes = f[args['semi_sup_idxes']][:]
            print("semi-supervised sentiment idxes:", sentiment_train_idxes.shape)
            senti_mask[sentiment_train_idxes] = 1.

    print(senti_mask.size())

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


    e2e = args['e2e']
    if not e2e:
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
                    norm=args['norm'], frozen_weights=args['freeze_weights'],
                    unimodal=args['unimodal']).to(device)

            print("Training one at a time...")

            lr = args['lr']

            N_EPOCHS = args['n_epochs']

            train_embed, (train_losses, valid_losses) = optimize_latents(args, True, gen_model, train_embedding,
                    dataloader, N_EPOCHS, lr, get_word_log_prob2, device,
                    validation_data=(valid_embedding, valid_dataloader))

            with open(os.path.join(folder, 'embed_loss.txt'), 'w') as f:
                for loss in train_losses:
                    f.write('{}\n'.format(loss))
            with open(os.path.join(folder, 'embed_valid_loss.txt'), 'w') as f:
                for loss in valid_losses:
                    f.write('{}\n'.format(loss))

            # if word2ix is not None:
            #     post_closest = get_closest_words(train_embed[:, :300].cpu().numpy(),
            #             word_embeddings.cpu().numpy(), word2ix)

            # with open(os.path.join(folder, 'closest_words.txt'), 'w') as f:
            #     for pre, post in zip(pre_closest, post_closest):
            #         f.write('{}\t{}\n'.format(' '.join(pre), ' '.join(post)))

            valid_embed, _ = optimize_latents(args, False, gen_model, valid_embedding,
                    valid_dataloader, N_EPOCHS, lr, get_word_log_prob2, device)
            test_embed, test_losses = optimize_latents(args, False, gen_model, test_embedding,
                    test_dataloader, N_EPOCHS, lr, get_word_log_prob2, device)

            with open(os.path.join(folder, 'embed_test_loss.txt'), 'w') as f:
                for loss in test_losses:
                    f.write('{}\n'.format(loss))

            torch.save(torch.cat([train_embed, valid_embed, test_embed], dim=0),
                    os.path.join(post_path, 'embed.bin'))

            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("Initial sentiment predictions, AFTER optimizing audio and visual")
            latents = (train_embed, valid_embed, test_embed)
            train_sentiment_for_latents(args, latents, sentiment_data, device,
                    train_idxes=sentiment_train_idxes,
                    model_save_path=post_path)
    else:
        # end-to-end training of latents
        print("end-to-end training of latents")

        # prepare sentiment data
        train_s, valid_s, test_s = sentiment_data

        senti_train_data = SentimentData(train_s, device)
        senti_valid_data = SentimentData(valid_s, device)
        senti_test_data = SentimentData(test_s, device)

        # all_train_data = ConcatDataset([train_dataset, senti_train_data])
        # all_valid_data = ConcatDataset([valid_dataset, senti_valid_data])
        # all_test_data = ConcatDataset([test_dataset, senti_test_data])

        # all_train_loader = DataLoader(all_train_data, batch_size=32, shuffle=True)
        # all_valid_loader = DataLoader(all_valid_data, batch_size=32, shuffle=True)
        # all_test_loader = DataLoader(all_test_data, batch_size=32, shuffle=True)

        senti_train_loader = DataLoader(senti_train_data, batch_size=32, shuffle=True)
        senti_valid_loader = DataLoader(senti_valid_data, batch_size=32, shuffle=True)
        senti_test_loader = DataLoader(senti_test_data, batch_size=32, shuffle=True)
        
        for r in range(args['n_runs']):
            if args['config_name']:
                config_name = args['config_name']
            else:
                config_name = os.path.split(os.path.split(args['config_file'])[0])[1]
            
            folder = 'model_saves/{}/config_{}_run_{}'.format(config_name, args['config_num'], r)
            if not os.path.isdir(folder):
                os.makedirs(folder)

            json.dump(args, open(os.path.join(folder, 'config.json'), 'w'), indent=2)
            
            pre_path = os.path.join(folder, 'pre')
            post_path = os.path.join(folder, 'post')

            if not os.path.isdir(pre_path):
                os.mkdir(pre_path)
            if not os.path.isdir(post_path):
                os.mkdir(post_path)

            torch.save(torch.tensor(combined_embedding.copy(), device=device, dtype=torch.float32),
                    os.path.join(pre_path, 'embed.bin'))

            # make generative model and sentiment model
            gen_model = AudioVisualGeneratorMultimodal(EMBEDDING_DIM, AUDIO_DIM, VISUAL_DIM,
                    norm=args['norm'], frozen_weights=args['freeze_weights'],
                    unimodal=args['unimodal']).to(device)

            if train['label'].ndim == 1:
                n_out = 1
            else:
                n_out = train['label'].shape[-1]

            senti_model = SentimentModel(EMBEDDING_DIM, args['sentiment_hidden_size'], n_out).to(device)

            # each iteration:
            # get likelihood and sentiment loss, add (weighted), and minimize

            train_embed = torch.tensor(train_embedding.copy(), device=device, dtype=torch.float32)
            train_embed.requires_grad = True

            grad_params = [train_embed]
            grad_params.extend(gen_model.parameters())
            grad_params.extend(senti_model.parameters())

            lr = args['lr']
            if args['optimizer'] == 'sgd':
                optimizer = optim.SGD(grad_params, lr=lr)
            elif args['optimizer'] == 'adam':
                optimizer = optim.Adam(grad_params, lr=lr)

            loss_function = nn.L1Loss(reduce=False)

            start_time = time.time()
            valid_niter = 10
            train_losses = []
            all_valid_losses = []
            N_EPOCHS = args['n_epochs']
            #N_EPOCHS = 10

            for i in range(N_EPOCHS):
                epoch_loss = 0.
                iters = 0

                for x in dataloader:
                    if args['dataset'] == 'mosi':
                        j, text, aud, vis, text_m, aud_m, vis_m, text_w = x
                    else:
                        j, text, aud, vis, text_m, aud_m, vis_m, text_w, text_a, text_a_m = x

                    _, s_data = senti_train_data[j]

                    iters += 1
                    optimizer.zero_grad()
                    out = gen_model(train_embed[j])

                    for modality, d in out.items():
                        if d['sigma'].min().abs() < 1e-7:
                            print(d, "boo!")

                    if args['dataset'] == 'mosi':
                        text_gauss = text
                        text_gauss_m = text_m
                    else:
                        text_gauss = text_a
                        text_gauss_m = text_a_m

                    if not args['unimodal']:
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

                    else:
                        batch_data = {
                            'text': text,
                            'audio': aud,
                            'visual': vis,
                            'text_weights': text_w,
                        }

                        batch_masks = {
                            'text': text_m,
                            'audio': aud_m,
                            'visual': vis_m,
                        }

                    log_prob = -get_log_prob_matrix(args, train_embed[j], out,
                            batch_data, batch_masks, get_word_log_prob2,
                            device=device, verbose=False)

                    # get sentiment accuracy
                    senti_predict = senti_model(train_embed[j])

                    senti_loss = loss_function(senti_predict, s_data)
                    # zero out the unsup idxes
                    if sentiment_train_idxes is not None:
                        mask = senti_mask[j]
                        senti_loss *= mask
                        
                    senti_loss = senti_loss.mean(dim=-1)

                    loss = args['likelihood_weight'] * log_prob + (1 - args['likelihood_weight']) * senti_loss
                    loss.mean().backward()
                    epoch_loss += float(loss.mean())

                    optimizer.step()

                train_losses.append(epoch_loss)
                if i % valid_niter == 0:
                    print("epoch {}: {} ({}s)".format(i, epoch_loss / iters, time.time() - start_time))

                    if i % (valid_niter * 8) == 0:
                        valid_embed, (valid_losses, _) = optimize_latents(args, False, gen_model, valid_embedding,
                                valid_dataloader, N_EPOCHS, lr, get_word_log_prob2, device, verbose=False)
                        print("Validation loss:", valid_losses[-1])
                        all_valid_losses.append(valid_losses[-1])

            # get test embeddings and evaluate them
            valid_embed, _ = optimize_latents(args, False, gen_model, valid_embedding,
                    valid_dataloader, N_EPOCHS, lr, get_word_log_prob2, device)
            test_embed, (test_losses, _) = optimize_latents(args, False, gen_model, test_embedding,
                    test_dataloader, N_EPOCHS, lr, get_word_log_prob2, device)

            if args['time_test']:
                # obtain test embeddings in closed form, get time taken
                # test_data = (test['text_id'], test['covarep'], test['facet'])
                # test_data = (
                #     torch.as_tensor(test['text_id'], dtype=torch.long, device=device),
                #     torch.as_tensor(test['covarep'], dtype=torch.float, device=device),
                #     torch.as_tensor(test['facet'], dtype=torch.float, device=device)
                # )

                print(list(test.keys()))
                print(list(test_mask.keys()))

                test_data = {
                    'text': torch.as_tensor(test['text'], dtype=torch.float, device=device),
                    'audio': torch.as_tensor(test['covarep'], dtype=torch.float, device=device),
                    'visual': torch.as_tensor(test['facet'], dtype=torch.float, device=device),
                }
                test_data.update({
                    'audiovisual': torch.cat([test_data['audio'], test_data['visual']], dim=-1),
                    'textaudio': torch.cat([test_data['text'], test_data['audio']], dim=-1),
                    'textvisual': torch.cat([test_data['text'], test_data['visual']], dim=-1),
                    'textaudiovisual': torch.cat([test_data['text'], test_data['audio'], test_data['visual']], dim=-1),
                })

                test_masks = {
                    'text': torch.as_tensor(test_mask['text'], dtype=torch.float, device=device),
                    'audio': torch.as_tensor(test_mask['covarep'], dtype=torch.float, device=device),
                    'visual': torch.as_tensor(test_mask['facet'], dtype=torch.float, device=device),
                }
                test_masks.update({
                    'audiovisual': torch.cat([test_masks['audio'], test_masks['visual']], dim=-1),
                    'textaudio': torch.cat([test_masks['text'], test_masks['audio']], dim=-1),
                    'textvisual': torch.cat([test_masks['text'], test_masks['visual']], dim=-1),
                    'textaudiovisual': torch.cat([test_masks['text'], test_masks['audio'], test_masks['visual']], dim=-1),
                })

                keys = [
                    'audio',
                    'visual',
                    'audiovisual',
                    'textaudio',
                    'textvisual',
                    'textaudiovisual',
                ]

                networks = {
                    k: (gen_model.embed2out[k]['mu'], gen_model.embed2out[k]['log_sigma'])
                    for k in keys
                }

                # audio_network = (gen_model.embed2out['audio']['mu'], gen_model.embed2out['audio']['log_sigma'])
                # visual_network = (gen_model.embed2out['visual']['mu'], gen_model.embed2out['visual']['log_sigma'])

                # get weights of text data
                text_tmp = torch.as_tensor(test['text_id'], dtype=torch.long, device=device)
                sentence_weights = torch.zeros_like(text_tmp, dtype=torch.float, device=device)
                mask = torch.ones_like(text_tmp)
                selection_arr = (mask > 0) & (text_tmp >= 0)

                for i in range(text_tmp.size()[0]):
                    sentence_weights[i] = torch.gather(weights, 0, text_tmp[i])
                sentence_weights = sentence_weights * selection_arr.type(torch.float)

                embeddings = word_embeddings[text_tmp,:]

                start_time = time.time()
                with torch.no_grad():
                    latents = estimate_embedding_overall_gpu2(test_data, test_masks, networks, sentence_weights, embeddings)
                    #latents = estimate_embedding_overall_gpu(test_data, test_mask, audio_network, visual_network,
                    #    sentence_weights, embeddings)
                end_time = time.time()

                print("time taken:", end_time - start_time)

                print("#############################################")
                print(test.keys())
                print(test_mask.keys())

                print(train_embed.size())
                print(valid_embed.size())
                print(test_embed.size())
                sys.exit()

            with open(os.path.join(folder, 'embed_loss.txt'), 'w') as f:
                for loss in train_losses:
                    f.write('{}\n'.format(loss))
            with open(os.path.join(folder, 'embed_valid_loss.txt'), 'w') as f:
                for loss in all_valid_losses:
                    f.write('{}\n'.format(loss))
            with open(os.path.join(folder, 'embed_test_loss.txt'), 'w') as f:
                for loss in test_losses:
                    f.write('{}\n'.format(loss))

            torch.save(torch.cat([train_embed, valid_embed, test_embed], dim=0),
                    os.path.join(post_path, 'embed.bin'))

            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("Initial sentiment predictions, AFTER optimizing audio and visual")
            train_embed.requires_grad = False
            valid_embed.requires_grad = False
            test_embed.requires_grad = False
            latents = (train_embed, valid_embed, test_embed)
            train_sentiment_for_latents(args, latents, sentiment_data, device,
                    train_idxes=sentiment_train_idxes,
                    model_save_path=post_path)

            sys.stdout.flush()

    sys.stdout.flush()

if __name__ == '__main__':
    main()
