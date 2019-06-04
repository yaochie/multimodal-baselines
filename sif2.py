import os
import argparse
import warnings
import time
import json
import pprint

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sif import load_weights, get_sentence_embeddings, get_sentence_word_weights
from losses import get_log_prob_matrix, get_word_log_prob_angular, get_word_log_prob_dot_prod
from models import AudioVisualGenerator
from sentiment_model import train_sentiment_for_latents
from utils import load_data, normalize_data, MMData

def estimate_embedding(data, W_mean, b_mean, W_log_sigma, b_log_sigma):
    """
    Estimate context vector given normally distributed data and
    the weight + bias for linear transformation of context
    to gaussian params

    data: (n_ex, len, n_features)
    W_mean, W_log_sigma: (n_features, context_size)
    b_mean, b_log_sigma: (n_features,)

    Return: (n_ex, context_size)
    """

    # handle differing sequence lengths?
    seq_len = data.shape[1]
    
    b_mean = b_mean.reshape((1, 1, -1))
    b_sigma = np.exp(b_log_sigma).reshape((1, 1, -1))

    q_mean = (data - b_mean) / (b_sigma ** 2)
    q_sigma = (data - b_sigma) ** 2 / (b_sigma ** 3) - 1. / b_sigma

    cs_mean = np.dot(q_mean, W_mean)
    cs_sigma = np.dot(q_sigma, np.exp(W_log_sigma))

    cs = (cs_mean.sum(axis=1) + cs_sigma.sum(axis=1)) / (2 * seq_len)
    
    return cs

def calc_weights2(data, b_mean, b_log_sigma):
    b_mean = b_mean.reshape((1, 1, -1))
    b_sigma = np.exp(b_log_sigma).reshape((1, 1, -1))

    q_mean = (data - b_mean) / (b_sigma ** 2)
    q_sigma = (data - b_mean) ** 2 / (b_sigma ** 3) - 1. / b_sigma
    return q_mean, q_sigma

def estimate_embedding_overall2(data, audio_networks, visual_networks, weights,
        word_embeddings):
    text, audio, visual = data
    audio_mu, audio_log_sigma = audio_networks
    visual_mu, visual_log_sigma = visual_networks

    q_mean_audio, q_sigma_audio = calc_weights(audio, audio_mu.bias.detach().cpu().numpy(),
            audio_log_sigma.bias.detach().cpu().numpy())
    q_mean_visual, q_sigma_visual = calc_weights(visual, visual_mu.bias.detach().cpu().numpy(),
            visual_log_sigma.bias.detach().cpu().numpy())

    W_mean_audio = audio_mu.weight.detach().cpu().numpy()
    W_sigma_audio = np.exp(audio_log_sigma.weight.detach().cpu().numpy())
    W_mean_visual = visual_mu.weight.detach().cpu().numpy()
    W_sigma_visual = np.exp(visual_log_sigma.weight.detach().cpu().numpy())

    # get weights of text data
    sentence_weights = get_sentence_word_weights(text, weights)

    # get total weight
    total_weight = sentence_weights.sum(-1) + q_mean_audio.sum(-1).sum(-1) + q_sigma_audio.sum(-1).sum(-1)
    total_weight += q_mean_visual.sum(-1).sum(-1) + q_sigma_visual.sum(-1).sum(-1)
    total_weight = total_weight.reshape((-1, 1, 1))
    
    q_mean_audio_norm = q_mean_audio / total_weight
    q_mean_visual_norm = q_mean_visual / total_weight
    q_sigma_audio_norm = q_sigma_audio / total_weight
    q_sigma_visual_norm = q_sigma_visual / total_weight
    sent_weight_norm = sentence_weights / total_weight.reshape((-1, 1))

    n_samples = sentence_weights.shape[0]
    cs = np.zeros((n_samples, 300))

    for i in range(n_samples):
        word_embeddings[text[i,:],:]
        cs[i,:] += sent_weight_norm[i,:].dot(word_embeddings[text[i,:],:])

    cs += np.dot(q_mean_audio_norm, W_mean_audio).sum(1)
    cs += np.dot(q_sigma_audio_norm, W_sigma_audio).sum(1)
    cs += np.dot(q_mean_visual_norm, W_mean_visual).sum(1)
    cs += np.dot(q_sigma_visual_norm, W_sigma_visual).sum(1)

    # normalize to get unit length vector
    return cs

def calc_weights(data, b_mean, b_log_sigma, mask):
    b_mean = b_mean.reshape((1, 1, -1))
    b_log_sigma = b_log_sigma.reshape((1, 1, -1))

    #q_mean = mask * (data - b_mean) / (np.exp(2 * b_log_sigma))
    #q_sigma = mask * (data - b_mean) ** 2 / np.exp(2 * b_log_sigma) - 1.
    #q_mean = (data - b_mean) / (np.exp(2 * b_log_sigma))
    #q_sigma = (data - b_mean) ** 2 / np.exp(2 * b_log_sigma) - 1.
    q_mean = (data - b_mean) / (torch.exp(2 * b_log_sigma))
    q_sigma = (data - b_mean) ** 2 / torch.exp(2 * b_log_sigma) - 1.

    return q_mean, q_sigma

def estimate_embedding_overall3(data, masks, audio_networks, visual_networks, sentence_weights,
        word_embeddings):
    text, audio, visual = data
    text_mask, audio_mask, visual_mask = masks
    audio_mu, audio_log_sigma = audio_networks
    visual_mu, visual_log_sigma = visual_networks

    q_mean_audio, q_sigma_audio = calc_weights(audio, audio_mu.bias.detach().cpu().numpy(),
            audio_log_sigma.bias.detach().cpu().numpy(), audio_mask)
    q_mean_visual, q_sigma_visual = calc_weights(visual, visual_mu.bias.detach().cpu().numpy(),
            visual_log_sigma.bias.detach().cpu().numpy(), visual_mask)

    W_mean_audio = audio_mu.weight.detach().cpu().numpy()
    W_log_sigma_audio = audio_log_sigma.weight.detach().cpu().numpy()
    W_mean_visual = visual_mu.weight.detach().cpu().numpy()
    W_log_sigma_visual = visual_log_sigma.weight.detach().cpu().numpy()

    # get weights of text data
    # sentence_weights = get_sentence_word_weights(text, weights)

    # get total weight
    total_weight = sentence_weights.sum(-1) + q_mean_audio.sum(-1).sum(-1) + q_sigma_audio.sum(-1).sum(-1)
    total_weight += q_mean_visual.sum(-1).sum(-1) + q_sigma_visual.sum(-1).sum(-1)
    total_weight = total_weight.reshape((-1, 1, 1))
    
    q_mean_audio_norm = q_mean_audio / total_weight
    q_mean_visual_norm = q_mean_visual / total_weight
    q_sigma_audio_norm = q_sigma_audio / total_weight
    q_sigma_visual_norm = q_sigma_visual / total_weight
    sent_weight_norm = sentence_weights / total_weight.reshape((-1, 1))

    n_samples = sentence_weights.shape[0]
    cs = np.zeros((n_samples, 300))

    for i in range(n_samples):
        # word_embeddings[text[i,:],:]
        cs[i,:] += sent_weight_norm[i,:].dot(word_embeddings[text[i,:],:])

    cs += np.dot(q_mean_audio_norm, W_mean_audio).sum(1)
    cs += np.dot(q_sigma_audio_norm, W_log_sigma_audio).sum(1)
    cs += np.dot(q_mean_visual_norm, W_mean_visual).sum(1)
    cs += np.dot(q_sigma_visual_norm, W_log_sigma_visual).sum(1)

    # normalize to get unit length vector
    cs /= np.linalg.norm(cs)

    return cs

def estimate_embedding_overall_gpu2(data, masks, networks, sentence_weights,
        embeddings):

    keys = [
        'audio',
        'visual',
        'audiovisual',
        'textaudio',
        'textvisual',
        'textaudiovisual',
    ]

    q_mean = {}
    q_sigma = {}
    W_mean = {}
    W_log_sigma = {}

    for k in keys:
        q_mean[k], q_sigma[k] = calc_weights(data[k], networks[k][0].bias, networks[k][1].bias, masks[k])
        W_mean[k] = networks[k][0].weight
        W_log_sigma[k] = networks[k][1].weight

    total_weight = sentence_weights.sum(-1) + sum(d.sum(-1).sum(-1) for d in q_mean.values())
    total_weight += sum(d.sum(-1).sum(-1) for d in q_sigma.values())
    total_weight = total_weight.reshape((-1, 1, 1))

    q_mean_norm = {}
    q_sigma_norm = {}
    for k in keys:
        q_mean_norm[k] = q_mean[k] / total_weight
        q_sigma_norm[k] = q_sigma[k] / total_weight

    sent_weight_norm = sentence_weights / total_weight.reshape((-1, 1))

    n_samples = sentence_weights.shape[0]

    cs = sent_weight_norm.unsqueeze(1).matmul(embeddings)
    cs = cs.squeeze()
    
    for k in keys:
        cs += torch.matmul(q_mean_norm[k], W_mean[k].unsqueeze(0)).sum(dim=1).squeeze()
        cs += torch.matmul(q_sigma_norm[k], W_log_sigma[k].unsqueeze(0)).sum(dim=1).squeeze()

    cs /= cs.norm(dim=1, keepdim=True)
    return cs

def estimate_embedding_overall_gpu(data, masks, audio_networks, visual_networks, sentence_weights,
        embeddings):
    text, audio, visual = data
    text_mask, audio_mask, visual_mask = masks
    audio_mu, audio_log_sigma = audio_networks
    visual_mu, visual_log_sigma = visual_networks

    print(audio.device)
    print(audio_mu.bias.device)

    q_mean_audio, q_sigma_audio = calc_weights(audio, audio_mu.bias,
            audio_log_sigma.bias, audio_mask)
    q_mean_visual, q_sigma_visual = calc_weights(visual, visual_mu.bias,
            visual_log_sigma.bias, visual_mask)

    W_mean_audio = audio_mu.weight
    W_log_sigma_audio = audio_log_sigma.weight
    W_mean_visual = visual_mu.weight
    W_log_sigma_visual = visual_log_sigma.weight

    # # get weights of text data
    # sentence_weights = torch.zeros_like(text, dtype=torch.float, device=audio.device)
    # mask = torch.ones_like(text)
    # selection_arr = (mask > 0) & (text >= 0)

    # for i in range(text.size()[0]):
    #     sentence_weights[i] = torch.gather(weights, 0, text[i])
    # #sentence_weights = torch.gather(weights, 0, text)

    # sentence_weights = sentence_weights * selection_arr.type(torch.float)

    # for i in range(text.size()[0]):
    #     for j in range(text.size()[1]):
    #         if mask[i, j] > 0 and text[i, j] >= 0:
    #             sentence_weights[i, j] = weights[text[i, j]]

    # sentence_weights = get_sentence_word_weights(text, weights)

    # get total weight
    total_weight = sentence_weights.sum(-1) + q_mean_audio.sum(-1).sum(-1) + q_sigma_audio.sum(-1).sum(-1)
    total_weight += q_mean_visual.sum(-1).sum(-1) + q_sigma_visual.sum(-1).sum(-1)
    total_weight = total_weight.reshape((-1, 1, 1))
    
    q_mean_audio_norm = q_mean_audio / total_weight
    q_mean_visual_norm = q_mean_visual / total_weight
    q_sigma_audio_norm = q_sigma_audio / total_weight
    q_sigma_visual_norm = q_sigma_visual / total_weight
    sent_weight_norm = sentence_weights / total_weight.reshape((-1, 1))

    n_samples = sentence_weights.shape[0]

    cs = sent_weight_norm.unsqueeze(1).matmul(embeddings)
    #cs = sent_weight_norm.unsqueeze(1).matmul(word_embeddings[text,:])
    cs = cs.squeeze()

    # for i in range(n_samples):
    #     # word_embeddings[text[i,:],:]
    #     cs[i,:] += sent_weight_norm[i,:].matmul(word_embeddings[text[i,:],:])

    cs += torch.matmul(q_mean_audio_norm, W_mean_audio.unsqueeze(0)).sum(dim=1).squeeze()
    cs += torch.matmul(q_sigma_audio_norm, W_log_sigma_audio.unsqueeze(0)).sum(dim=1).squeeze()
    cs += torch.matmul(q_mean_visual_norm, W_mean_visual.unsqueeze(0)).sum(dim=1).squeeze()
    cs += torch.matmul(q_sigma_visual_norm, W_log_sigma_visual.unsqueeze(0)).sum(dim=1).squeeze()

    # normalize to get unit length vector
    cs /= cs.norm(dim=1, keepdim=True)

    return cs

def estimate_embedding_overall(data, masks, audio_networks, visual_networks, weights,
        word_embeddings):
    text, audio, visual = data
    text_mask, audio_mask, visual_mask = masks
    audio_mu, audio_log_sigma = audio_networks
    visual_mu, visual_log_sigma = visual_networks

    q_mean_audio, q_sigma_audio = calc_weights(audio, audio_mu.bias.detach().cpu().numpy(),
            audio_log_sigma.bias.detach().cpu().numpy(), audio_mask)
    q_mean_visual, q_sigma_visual = calc_weights(visual, visual_mu.bias.detach().cpu().numpy(),
            visual_log_sigma.bias.detach().cpu().numpy(), visual_mask)

    W_mean_audio = audio_mu.weight.detach().cpu().numpy()
    W_log_sigma_audio = audio_log_sigma.weight.detach().cpu().numpy()
    W_mean_visual = visual_mu.weight.detach().cpu().numpy()
    W_log_sigma_visual = visual_log_sigma.weight.detach().cpu().numpy()

    # get weights of text data
    sentence_weights = get_sentence_word_weights(text, weights)

    # get total weight
    total_weight = sentence_weights.sum(-1) + q_mean_audio.sum(-1).sum(-1) + q_sigma_audio.sum(-1).sum(-1)
    total_weight += q_mean_visual.sum(-1).sum(-1) + q_sigma_visual.sum(-1).sum(-1)
    total_weight = total_weight.reshape((-1, 1, 1))
    
    q_mean_audio_norm = q_mean_audio / total_weight
    q_mean_visual_norm = q_mean_visual / total_weight
    q_sigma_audio_norm = q_sigma_audio / total_weight
    q_sigma_visual_norm = q_sigma_visual / total_weight
    sent_weight_norm = sentence_weights / total_weight.reshape((-1, 1))

    n_samples = sentence_weights.shape[0]
    cs = np.zeros((n_samples, 300))

    for i in range(n_samples):
        # word_embeddings[text[i,:],:]
        cs[i,:] += sent_weight_norm[i,:].dot(word_embeddings[text[i,:],:])

    cs += np.dot(q_mean_audio_norm, W_mean_audio).sum(1)
    cs += np.dot(q_sigma_audio_norm, W_log_sigma_audio).sum(1)
    cs += np.dot(q_mean_visual_norm, W_mean_visual).sum(1)
    cs += np.dot(q_sigma_visual_norm, W_log_sigma_visual).sum(1)

    # normalize to get unit length vector
    cs /= np.linalg.norm(cs)

    return cs

def estimate_embedding_wrapper(data, linear_layers):
    mean_linear_layer = linear_layers['mu']
    log_sigma_linear_layer = linear_layers['log_sigma']

    W_mean = mean_linear_layer.weight.detach().cpu().numpy()
    b_mean = mean_linear_layer.bias.detach().cpu().numpy()

    W_log_sigma = log_sigma_linear_layer.weight.detach().cpu().numpy()
    b_log_sigma = log_sigma_linear_layer.bias.detach().cpu().numpy()

    return estimate_embedding(data, W_mean, b_mean, W_log_sigma, b_log_sigma)

def optimize_embeddings(args, text_embeddings, data, masks, weight,
        word_embeddings, weights, dataloader, device):
    """
    Arguments:
        text_embeddings: sentence embeddings from text modality
        audio: audio data
        visual: visual data
        weight: relative weights of the embeddings of each modality

    Returns:
        final data
    """

    text, audio, visual = data
    text_mask = masks['text']
    audio_mask = masks['covarep']
    visual_mask = masks['facet']

    EMBEDDING_DIM = text_embeddings.shape[-1]
    AUDIO_DIM = audio.shape[-1]
    VISUAL_DIM = visual.shape[-1]
    
    if len(weight) != 3:
        raise ValueError("Invalid weight")

    if sum(weight) != 1:
        warnings.warn("weight doesn't sum to 1, normalizing..")
        total = sum(weight)
        weight = [float(x) / total for x in weight]

    if args['word_sim_metric'] == 'angular':
        word_log_prob_fn = get_word_log_prob_angular
    elif args['word_sim_metric'] == 'dot_prod':
        word_log_prob_fn = get_word_log_prob_dot_prod
    else:
        raise NotImplementedError

    a = 1e-3

    def get_word_log_prob(latents, text_data, mask):
        word_log_prob = word_log_prob_fn(latents, weights, word_embeddings, text_data, mask, a)
        if word_log_prob.min().abs() == np.inf:
            print('word inf')
            print(latents.size())
            print(latents.matmul(word_embeddings.transpose(0, 1)).max())
            print(latents.matmul(word_embeddings.transpose(0, 1)).exp().max())
            print(latents)
            sys.exit()

        return word_log_prob

    # create network
    network = AudioVisualGenerator(EMBEDDING_DIM, AUDIO_DIM, VISUAL_DIM, frozen_weights=False).to(device)

    lr = args['lr']
    optimizer = optim.SGD(network.parameters(), lr=lr)

    if args.get('autoscale'):
        n_examples = 1000
        word_mult = torch.ones(n_examples, requires_grad=True, device=device)
        audio_mult = torch.ones(n_examples, requires_grad=True, device=device)
        visual_mult = torch.ones(n_examples, requires_grad=True, device=device)

        mult_optimizer = optim.SGD([word_mult, audio_mult, visual_mult], lr=lr)

    start_time = time.time()
    N_EPOCHS = 100
    train_losses = []
    for i in range(N_EPOCHS):
        epoch_loss = 0.

        # estimate cs from word, audio and visual, and weight them.
        # text cs is fixed, so we just estimate the other two.

        # audio_embeddings = estimate_embedding_wrapper(audio,
        #                                              network.embed2audio)
        # visual_embeddings = estimate_embedding_wrapper(visual,
        #                                               network.embed2visual)

        estimate = estimate_embedding_overall((text, audio, visual),
                (text_mask, audio_mask, visual_mask),
                (network.embed2audio['mu'], network.embed2audio['log_sigma']),
                (network.embed2visual['mu'], network.embed2visual['log_sigma']),
                weights, word_embeddings)
        estimate = torch.tensor(estimate, dtype=torch.float, device=device)
        # estimate = estimate / estimate.norm()

        # estimate = torch.tensor(weight[0] * text_embeddings +
        #             weight[1] * audio_embeddings +
        #             weight[2] * visual_embeddings, dtype=torch.float, device=device)

        # Now given cs, calculate log probability of the data, and optimize
        iters = 0
        for j, txt, aud, vis, text_m, aud_m, vis_m in dataloader:
            iters += 1
            optimizer.zero_grad()
            audio_p, visual_p = network(estimate[j])
            
            log_prob = -get_log_prob_matrix(args, estimate[j], audio_p, visual_p,
                    {"text": txt, "covarep": aud, "facet": vis},
                    {"text": text_m, "covarep": aud_m, "facet": vis_m},
                    get_word_log_prob,
                    device=device)

            avg_log_prob = log_prob.mean()
            avg_log_prob.backward()

            optimizer.step()
            epoch_loss += avg_log_prob

        train_losses.append(epoch_loss)
        print("epoch {}: {} ({}s)".format(i, epoch_loss / iters, time.time() - start_time))

        # normalize latents??

    # return final estimate of optimal embeddings
    estimate = estimate_embedding_overall((text, audio, visual),
            (text_mask, audio_mask, visual_mask),
            (network.embed2audio['mu'], network.embed2audio['log_sigma']),
            (network.embed2visual['mu'], network.embed2visual['log_sigma']),
            weights, word_embeddings)
    estimate = torch.tensor(estimate, dtype=torch.float, device=device)

    return estimate, train_losses

def update_masks(mask_dict, data):
    mask_dict['text'] = (data != 0).astype(int)

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
    parser.add_argument('--cuda_device', type=int, choices=list(range(4)))
    parser.add_argument('--semi_sup_idxes', choices=['{:.1f}'.format(x) for x in np.arange(0.1, 1, 0.1)])
    parser.add_argument('--config_name', help='override config name in config file')
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

    word2ix, word_embeddings, data = load_data()
    train, valid, test = data

    train, train_mask = normalize_data(train)
    valid, valid_mask = normalize_data(valid)
    test, test_mask = normalize_data(test)

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
    combined_data = (
        combined_text,
        combined_covarep,
        combined_facet
    )

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
    dataset = MMData(combined_text, combined_covarep, combined_facet, combined_masks, device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    sentiment_data = (train['label'], valid['label'], test['label'])
    sentiment_train_idxes = None
    if args.get('semi_sup_idxes') is not None:
        with h5py.File('subset_idxes.h5', 'r') as f:
            sentiment_train_idxes = f[args['semi_sup_idxes']][:]
            print(sentiment_train_idxes.shape)

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

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("Initial sentiment predictions, before optimizing audio and visual")

        curr_embedding = torch.tensor(combined_embedding.copy(), device=device, dtype=torch.float32)

        train_sentiment_for_latents(args, curr_embedding, sentiment_data, device,
                (n_train, n_valid, n_test), train_idxes=sentiment_train_idxes,
                model_save_path=pre_path)

        torch.save(curr_embedding, os.path.join(pre_path, 'embed.bin'))

        embeddings, train_losses = optimize_embeddings(args, curr_embedding, combined_data, combined_masks,
                [1, 1, 1], word_embeddings, weights, dataloader, device)

        with open(os.path.join(folder, 'embed_loss.txt'), 'w') as f:
            for loss in train_losses:
                f.write('{}\n'.format(loss))
        torch.save(embeddings, os.path.join(post_path, 'embed.bin'))

        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Initial sentiment predictions, AFTER optimizing audio and visual")
        train_sentiment_for_latents(args, embeddings, sentiment_data, device,
                (n_train, n_valid, n_test), train_idxes=sentiment_train_idxes,
                model_save_path=post_path)

if __name__ == '__main__':
    main()
