import os
import argparse
import warnings
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sif import load_weights, get_word_embeddings, get_sentence_word_weights
from losses import get_log_prob_matrix, get_word_log_prob_angular, get_word_log_prob_dot_prod
from models import AudioVisualGenerator
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

def calc_weights(data, b_mean, b_log_sigma):
    b_mean = b_mean.reshape((1, 1, -1))
    b_log_sigma = b_log_sigma.reshape((1, 1, -1))

    q_mean = (data - b_mean) / (np.exp(2 * b_log_sigma))
    q_sigma = (data - b_mean) ** 2 / np.exp(2 * b_log_sigma) - 1.

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

def estimate_embedding_overall(data, audio_networks, visual_networks, weights,
        word_embeddings):
    text, audio, visual = data
    audio_mu, audio_log_sigma = audio_networks
    visual_mu, visual_log_sigma = visual_networks

    q_mean_audio, q_sigma_audio = calc_weights(audio, audio_mu.bias.detach().cpu().numpy(),
            audio_log_sigma.bias.detach().cpu().numpy())
    q_mean_visual, q_sigma_visual = calc_weights(visual, visual_mu.bias.detach().cpu().numpy(),
            visual_log_sigma.bias.detach().cpu().numpy())

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
        word_embeddings[text[i,:],:]
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

def optimize_embeddings(args, text_embeddings, data, weight,
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

    EMBEDDING_DIM = text_embeddings.shape[-1]
    AUDIO_DIM = audio.shape[-1]
    VISUAL_DIM = visual.shape[-1]
    
    if len(weight) != 3:
        raise ValueError("Invalid weight")

    if sum(weight) != 1:
        warnings.warn("weight doesn't sum to 1, normalizing..")
        total = sum(weight)
        weight = [float(x) / total for x in weight]

    args.update({
        'word_sim_metric': 'angular',
        'word_loss_weight': 0.01,
    })
    if args['word_sim_metric'] == 'angular':
        word_log_prob_fn = get_word_log_prob_angular
    elif args['word_sim_metric'] == 'dot_prod':
        word_log_prob_fn = get_word_log_prob_dot_prod
    else:
        raise NotImplementedError

    a = 1e-3

    def get_word_log_prob(latents, text_data):
        word_log_prob = word_log_prob_fn(latents, weights, word_embeddings, text_data, a)
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
        for j, txt, aud, vis in dataloader:
            iters += 1
            optimizer.zero_grad()
            audio_p, visual_p = network(estimate[j])
            
            log_prob = -get_log_prob_matrix(args, estimate[j], audio_p, visual_p,
                    {"text": txt, "covarep": aud, "facet": vis}, get_word_log_prob,
                    device=device, verbose=True)

            avg_log_prob = log_prob.mean()
            avg_log_prob.backward()

            optimizer.step()
            epoch_loss += avg_log_prob

        train_losses.append(epoch_loss)
        print("epoch {}: {} ({}s)".format(i, epoch_loss / iters, time.time() - start_time))

        # normalize latents??

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cuda_device', type=int, choices=list(range(4)))

    args = vars(parser.parse_args())

    return args

def main():
    args = parse_arguments()

    if args['cuda_device']:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args['cuda_device'])

    device = torch.device('cuda')

    word2ix, word_embeddings, data = load_data()
    train, valid, test = data

    train = normalize_data(train)
    valid = normalize_data(valid)
    test = normalize_data(test)

    n_train = train['label'].shape[0]
    n_valid = valid['label'].shape[0]
    n_test = test['label'].shape[0]

    combined_text = np.concatenate([train['text'], valid['text'], test['text']], axis=0)
    combined_covarep = np.concatenate([train['covarep'], valid['covarep'], test['covarep']], axis=0)
    combined_facet = np.concatenate([train['facet'], valid['facet'], test['facet']], axis=0)

    weights = load_weights()
    weights = torch.tensor(weights, device=device, dtype=torch.float32)
    word_embeddings = torch.tensor(word_embeddings, device=device, dtype=torch.float32)

    # if args['word_sim_metric'] == 'dot_prod':
    #     word_embeddings = F.normalize(word_embeddings)

    train_embedding = get_word_embeddings(word_embeddings, weights, train['text'])
    valid_embedding = get_word_embeddings(word_embeddings, weights, valid['text'])
    test_embedding = get_word_embeddings(word_embeddings, weights, test['text'])
    combined_embedding = np.concatenate([train_embedding, valid_embedding, test_embedding], axis=0)

    BATCH_SIZE = 64
    dataset = MMData(train['text'], train['covarep'], train['facet'], device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimize_embeddings(args, train_embedding, (train['text'], train['covarep'], train['facet']), [1, 1, 1],
            word_embeddings, weights, dataloader, device)

if __name__ == '__main__':
    main()
