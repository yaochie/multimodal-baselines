import sys

import torch
import torch.nn as nn

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import numpy as np

def get_normal_log_prob(mu, sigma, values, mask):
    """
    Arguments:
        mu: a (batch_size, 1, n_features) tensor of the mean of each feature
        sigma: (batch_size, 1, n_features) tensor of the stdev of each feature
        values: (batch_size, seq_len, n_features) tensor of the values of the feature

    Returns:
        A (batch_size,) tensor of the sum of the log probabilities of each sample,
        assuming each feature is independent and normally-distributed according to
        the given mu and sigma.
    """
    sig_sq = sigma.pow(2)
    term1 = torch.log(1. / torch.sqrt(2. * np.pi * sig_sq))

    diff = values - mu
    term2 = diff.pow(2) / (2. * sig_sq)

    log_prob = term1 - term2
    masked_log_prob = log_prob * mask
    return masked_log_prob.squeeze().sum(-1).sum(-1)
    #return log_prob.squeeze().sum(-1).sum(-1)

def get_word_log_prob_angular(latents, weights, word_embeddings, data, mask, a):
    """
    Calculate the log probability of the word data given the latents, using
    the angular distance between the latent embedding and the word embeddings
    of the sentence. Based on Ethayarajh's work.
    """
    coss = nn.CosineSimilarity(dim=-1)

    cosine_sims = coss(latents.unsqueeze(1), word_embeddings.unsqueeze(0))
    angular_dists = cosine_sims.acos()
    Z_s = (1. - angular_dists / np.pi).sum(-1, keepdim=True)
    #Z_s = latents.matmul(word_embeddings.transpose(0, 1)).exp().sum(-1, keepdim=True)
    alpha = 1. / (Z_s * a + 1.)

    word_weights = weights[data]
    sent_embeddings = word_embeddings[data]

    unigram_prob = alpha * word_weights

    score = 1. - (coss(sent_embeddings, latents.unsqueeze(1)).acos() / np.pi)
    context_prob = (1. - alpha) * score / Z_s
    #dot_prod = torch.bmm(sent_embeddings, latents.unsqueeze(-1)).squeeze()
    #context_prob = (1. - alpha) * dot_prod.exp() / Z_s

    log_probs = torch.log(unigram_prob + context_prob)

    # mask probabilities of padding words
    log_probs = log_probs * mask

    word_log_prob = log_probs.sum(dim=-1)
    return word_log_prob

def get_word_log_prob_angular2(latents, word_embeddings, word_weights, sent_embeddings, mask, a):
    """
    Calculate the log probability of the word data given the latents, using
    the angular distance between the latent embedding and the word embeddings
    of the sentence. Based on Ethayarajh's work.
    """
    coss = nn.CosineSimilarity(dim=-1)

    cosine_sims = coss(latents.unsqueeze(1), word_embeddings.unsqueeze(0))
    angular_dists = cosine_sims.acos()
    Z_s = (1. - angular_dists / np.pi).sum(-1, keepdim=True)
    #Z_s = latents.matmul(word_embeddings.transpose(0, 1)).exp().sum(-1, keepdim=True)
    alpha = 1. / (Z_s * a + 1.)

    unigram_prob = alpha * word_weights

    score = 1. - (coss(sent_embeddings, latents.unsqueeze(1)).acos() / np.pi)
    context_prob = (1. - alpha) * score / Z_s
    #dot_prod = torch.bmm(sent_embeddings, latents.unsqueeze(-1)).squeeze()
    #context_prob = (1. - alpha) * dot_prod.exp() / Z_s

    log_probs = torch.log(unigram_prob + context_prob)

    # mask probabilities of padding words
    log_probs = log_probs * mask[:,:,0]

    word_log_prob = log_probs.sum(dim=-1)
    return word_log_prob


def get_word_log_prob_dot_prod(latents, weights, word_embeddings, data, a):
    """
    Arora's original log probability formulation, based on dot product.
    ** sensitive to vector norm!
    exponentiate the sigma

    calculate probabilities for text using Arora model
    given word weights p(w), a / (a + p(w)) * v(w) . embedding

    use softmax instead 
    log (alpha * p(w) + (1 - alpha) exp(dotprod) / Z)
    calc partition value Z - sum of exps of inner products of embedding with all words.
    """
    Z_s = latents.matmul(word_embeddings.transpose(0, 1)).exp().sum(-1, keepdim=True)
    alpha = 1. / (Z_s * a + 1.)

    word_weights = weights[data]
    sent_embeddings = word_embeddings[data]

    unigram_prob = alpha * word_weights

    dot_prod = torch.bmm(sent_embeddings, latents.unsqueeze(-1)).squeeze()
    context_prob = (1. - alpha) * dot_prod.exp() / Z_s

    log_probs = torch.log(unigram_prob + context_prob)
    word_log_prob = log_probs.sum(dim=-1)
    return word_log_prob

def get_word_log_prob_dot_prod2(latents, word_embeddings, word_weights, sent_embeddings, mask, a):
    """
    Arora's original log probability formulation, based on dot product.
    ** sensitive to vector norm!
    exponentiate the sigma

    calculate probabilities for text using Arora model
    given word weights p(w), a / (a + p(w)) * v(w) . embedding

    use softmax instead 
    log (alpha * p(w) + (1 - alpha) exp(dotprod) / Z)
    calc partition value Z - sum of exps of inner products of embedding with all words.
    """
    Z_s = latents.matmul(word_embeddings.transpose(0, 1)).exp().sum(-1, keepdim=True)
    alpha = 1. / (Z_s * a + 1.)

    unigram_prob = alpha * word_weights

    dot_prod = torch.bmm(sent_embeddings, latents.unsqueeze(-1)).squeeze()
    context_prob = (1. - alpha) * dot_prod.exp() / Z_s

    log_probs = torch.log(unigram_prob + context_prob)
    log_probs = log_probs * mask[:,:,0]

    word_log_prob = log_probs.sum(dim=-1)
    return word_log_prob

def get_log_prob_matrix(args, latents, audio, visual, data, masks, word_log_prob_fn,
        device=torch.device('cpu'), verbose=False):
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
        masks: a binary tensor indicating whether the loss for this value
            should be masked (because it is padding)
    """
    epsilon = torch.tensor(1e-6, device=device)

    (audio_mu, audio_sigma) = audio
    (visual_mu, visual_sigma) = visual
    #audio_sigma = audio_sigma + epsilon
    #visual_sigma = visual_sigma + epsilon

    word_log_prob = word_log_prob_fn(latents, data['text'], masks['text'])

    """
    assume samples in sequences are i.i.d.
    calculate probabilities for audio and visual as if
    sampling from distribution

    audio: (batch, seqlength, n_features)
    audio_mu: (batch, n_features)
    audio_sigma: (batch, n_features)
    independent normals, so just calculate log prob directly
    """

    audio_log_prob = get_normal_log_prob(audio_mu.unsqueeze(1),
                audio_sigma.unsqueeze(1), data['covarep'], masks['covarep'])
    visual_log_prob = get_normal_log_prob(visual_mu.unsqueeze(1),
                visual_sigma.unsqueeze(1), data['facet'], masks['facet'])

    bad = False
    if audio_log_prob.min().abs() == np.inf:
        print('aud inf')
        bad = True
    if visual_log_prob.min().abs() == np.inf:
        print('vis inf')
        bad = True
    if bad:
        sys.exit()

    if verbose:
        print("Visual: {}\tAudio: {}\tWord: {}".format(visual_log_prob.min(),
            audio_log_prob.min(), word_log_prob.min()))

        # print("Visual: {}".format(visual_log_prob.min()))
        # print("Audio: {}".format(audio_log_prob.min()))
        # print("Word: {}".format(word_log_prob.min()))

    # final output: one value per datapoint
    if 'word_loss_weight' in args:
        word_weight = args['word_loss_weight']
        aud_weight = vis_weight = (1. - word_weight) / 2
        total_log_prob = aud_weight * audio_log_prob + vis_weight * visual_log_prob + word_weight * word_log_prob
    else:
        total_log_prob = audio_log_prob + visual_log_prob + word_log_prob
    # total_log_prob = audio_log_prob + visual_log_prob
    return total_log_prob

def get_log_prob_matrix_trimodal(args, latents, out, data, masks, word_log_prob_fn,
        device=torch.device('cpu'), verbose=False):
    """
    Return the log probability for the batch data given the
    latent variables, and the derived audio and visual parameters.

    Also have for multimodal (concatenation). Treat multimodal outputs as gaussian-distributed.

    Returns one log-probability value per example in the batch.

    Arguments:
        latents: the (joint) latent variables for the batch
        audio: the audio params for the batch (tuple mu, sigma)
        visual: the visual params for the batch (tuple mu, sigma)
        data: dict containing text, audio and visual features.
            text is a tensor of word ids, covarep is a tensor of audio
            features, facet is a tensor of visual features.
        masks: a binary tensor indicating whether the loss for this value
            should be masked (because it is padding)
    """
    word_log_prob = word_log_prob_fn(latents, data['text_weights'], data['text'], masks['text'])

    """
    assume samples in sequences are i.i.d.
    calculate probabilities for audio and visual as if
    sampling from distribution

    audio: (batch, seqlength, n_features)
    audio_mu: (batch, n_features)
    audio_sigma: (batch, n_features)
    independent normals, so just calculate log prob directly
    """

    log_probs = {}

    for modality, d in out.items():
        mu = d['mu']
        sigma = d['sigma']

        log_probs[modality] = get_normal_log_prob(mu.unsqueeze(1),
                sigma.unsqueeze(1), data[modality], masks[modality])

    bad = False
    for m, lp in log_probs.items():
        if lp.min().abs() == np.inf:
            print(m, 'inf')
            bad = True
    if bad:
        sys.exit()

    if verbose:
        print("Visual: {}\tAudio: {}\tWord: {}".format(visual_log_prob.min(),
            audio_log_prob.min(), word_log_prob.min()))

    # final output: one value per datapoint
    if 'word_loss_weight' in args:
        word_weight = args['word_loss_weight']
        other_weight = (1. - word_weight) / len(log_probs)
        total_log_prob = sum(log_probs.values()) * other_weight + word_weight * word_log_prob
    else:
        total_log_prob = sum(log_probs.values()) + word_log_prob
    # total_log_prob = audio_log_prob + visual_log_prob
    return total_log_prob

def full_loss(predictions, y_test):
    """
    predictions and y_test should be numpy matrices
    """

    predictions = predictions.flatten()
    y_test = y_test.flatten()
    #predictions = predictions.reshape((len(y_test),))

    mae = np.mean(np.absolute(predictions-y_test))
    print("mae: {}".format(mae))
    corr = np.corrcoef(predictions,y_test)[0][1]
    print("corr: {}".format(corr))
    mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
    print("mult_acc: {}".format(mult))
    f_score = round(f1_score(np.round(predictions),np.round(y_test),average='weighted'),5)
    print("mult f_score: {}".format(f_score))

    true_label = (y_test >= 0)
    predicted_label = (predictions >= 0)
    accuracy = accuracy_score(true_label, predicted_label)
    print("Confusion Matrix :")
    print(confusion_matrix(true_label, predicted_label))
    print("Classification Report :")
    print(classification_report(true_label, predicted_label, digits=5))
    print("Accuracy {}".format(accuracy))

    results = {
        'mae': float(mae),
        'accuracy': float(accuracy),
        'corr': float(corr),
        'mult_acc': float(mult),
        'f_score': float(f_score)
    }

    return results
