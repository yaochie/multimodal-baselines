import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from losses import (get_normal_log_prob, get_word_log_prob_angular,
        get_word_log_prob_dot_prod, full_loss)

class SentimentModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(SentimentModel, self).__init__()

        self.hidden1 = nn.Linear(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        x = F.relu(self.hidden1(inputs))
        x = self.out(x)
        # x = F.tanh(self.out(x)) * 3
        # sentiment is [-3, 3] range
        return x.squeeze()

def save_sentiment(path, model):
    torch.save(model.state_dict(), os.path.join(path, 'senti.bin'))

def load_sentiment(path, embedding_dim, hidden_dim, device):
    model = SentimentModel(embedding_dim, hidden_dim)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    return model

def eval_sentiment(data, model, latents):
    n_samples = len(data.dataset)
    loss_function = nn.L1Loss(reduce=False)
    total_loss = 0

    y_test = []
    predictions = []

    with torch.no_grad():
        for j, senti in data:
            senti_predict = model(latents[j])
            loss = loss_function(senti_predict, senti)
            total_loss += loss.sum()

            y_test.append(senti)
            predictions.append(senti_predict)

    print("MAE: {}".format(total_loss / n_samples))

    y_test = torch.cat(y_test)
    predictions = torch.cat(predictions)
    # print(y_test.size())
    return full_loss(predictions, y_test)

def train_sentiment(args, data, model, latents, valid_niter=10, verbose=False):
    n_epochs = args['n_sentiment_epochs']
    lr = args['sentiment_lr']
    if verbose:
        lr = 1e-6

    n_samples = len(data.dataset)
    loss_function = nn.L1Loss(reduce=False)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_losses = []
    for i in range(n_epochs):
        epoch_loss = 0
        for j, senti in data:
            model.zero_grad()
            senti_predict = model(latents[j])
            
            loss = loss_function(senti_predict, senti)
            epoch_loss += loss.sum()
            #if verbose:
            #    print(epoch_loss)
            #    print(loss.mean())

            loss.mean().backward()
            optimizer.step()

        train_losses.append(epoch_loss)
        #if verbose:
        #    sys.exit()
        if i % valid_niter == 0:
            print("Epoch {}: {}".format(i, epoch_loss / n_samples))
    print("Epoch {}: {}".format(i, epoch_loss / n_samples))
    return train_losses

def train_sentiment_for_latents(args, latents, sentiment_data, device, verbose=False,
            model_save_path=None):
    hidden_dim = args['sentiment_hidden_size']

    embedding_dim = latents.size()[-1]
    senti_model = SentimentModel(embedding_dim, hidden_dim).to(device)

    print("Initial sentiment predictions")
    senti_model.eval()
    acc = eval_sentiment(sentiment_data, senti_model, latents)
    if model_save_path is not None:
        with open(os.path.join(model_save_path, 'acc_before.txt'), 'w') as f:
            f.write(str(acc))

    print("Training sentiment model on sentence embeddings...")
    senti_model.train()
    train_losses = train_sentiment(args, sentiment_data, senti_model, latents, verbose=verbose)

    with open(os.path.join(model_save_path, 'senti_train_loss.txt'), 'w') as f:
        for loss in train_losses:
            f.write('{}\n'.format(loss))
    if model_save_path is not None:
        save_sentiment(model_save_path, senti_model)

    print("Sentiment predictions after training")
    senti_model.eval()
    acc = eval_sentiment(sentiment_data, senti_model, latents)
    if model_save_path is not None:
        with open(os.path.join(model_save_path, 'acc_after.txt'), 'w') as f:
            f.write(str(acc))
    print("-----------------------------")
