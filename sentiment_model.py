import sys
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from losses import (get_normal_log_prob, get_word_log_prob_angular,
        get_word_log_prob_dot_prod, full_loss, iemocap_loss, pom_loss)

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

class SentimentModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_out):
        super(SentimentModel, self).__init__()

        self.hidden1 = nn.Linear(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_out)

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

def predict_sentiment(data, model, latents):
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

    y_test = torch.cat(y_test).cpu().numpy()
    predictions = torch.cat(predictions).cpu().numpy()

    return predictions, y_test

def train_sentiment(args, model, train_data, train_latents,
            valid_data, valid_latents, model_loader, valid_niter=10,
            verbose=False, model_save_path=None):
    """
    Train sentiment model
    TODO: implement early stopping
    """

    n_epochs = args['n_sentiment_epochs']
    lr = args['sentiment_lr']
    patience = 10
    n_trials = 3

    n_samples = len(train_data.dataset)
    loss_function = nn.L1Loss(reduce=False)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_losses = []
    valid_losses = []
    n_bad = 0
    n_bad_trials = 0

    for i in range(n_epochs):
        epoch_loss = 0
        n_batches = 0
        for j, senti in train_data:
            n_batches += 1
            model.zero_grad()
            senti_predict = model(train_latents[j])
            
            loss = loss_function(senti_predict, senti)
            epoch_loss += loss.mean()

            loss.mean().backward()
            optimizer.step()
        
        avg_epoch_loss = epoch_loss / n_batches
        train_losses.append(avg_epoch_loss)
        if i % valid_niter == 0:
            batches = 0
            valid_loss = 0
            with torch.no_grad():
                for j, senti in valid_data:
                    senti_predict = model(valid_latents[j])
                    loss = loss_function(senti_predict, senti)

                    valid_loss += loss.mean()
                    batches += 1

            avg_valid_loss = valid_loss / batches
            print("Epoch {}: {} (avg val loss {})".format(i, epoch_loss / n_batches,
                    avg_valid_loss))

            is_better = len(valid_losses) == 0 or avg_valid_loss < min(valid_losses)
            valid_losses.append(avg_valid_loss)

            if args['early_stopping']:
                if is_better:
                    n_bad = 0
                    if model_save_path is not None:
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, os.path.join(model_save_path, 'senti.bin'))
                else:
                    print('patience {}'.format(n_bad))
                    n_bad += 1
                    if n_bad >= patience:
                        n_bad_trials += 1
                        if n_bad_trials < n_trials:

                            if model_save_path is not None:
                                print("reloading model and decaying learning rate...")
                                checkpoint = torch.load(os.path.join(model_save_path, 'senti.bin'))
                                model.load_state_dict(checkpoint['model_state_dict'])
                                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                            # decay learning rate
                            lr = lr * args['lr_decay']
                            for g in optimizer.param_groups:
                                g['lr'] = lr
                            n_bad = 0
                        else:
                            print("early stopping...")
                            break

    print("Epoch {}: {}".format(i, epoch_loss / n_samples))
    return train_losses, valid_losses

def train_sentiment_for_latents(args, latents, sentiment_data, device,
            verbose=False, model_save_path=None, train_idxes=None):

    train_latents, valid_latents, test_latents = latents
    # (n_train, n_valid, n_test) = counts
    hidden_dim = args['sentiment_hidden_size']

    embedding_dim = train_latents.size()[-1]

    # prepare data
    #train_latents = latents[:n_train]
    #valid_latents = latents[n_train:n_train+n_valid]
    #test_latents = latents[n_train+n_valid:]

    train, valid, test = sentiment_data

    if train.ndim == 1:
        n_out = 1
    else:
        n_out = train.shape[-1]
    senti_model = SentimentModel(embedding_dim, hidden_dim, n_out).to(device)

    print("train data shape:", train.shape)
    print("train latents shape:", train_latents.size())
    if train_idxes is not None:
        train = train[train_idxes]
        train_latents = train_latents[train_idxes]

        print("train data shape:", train.shape)
        print("train latents shape:", train_latents.size())

    train_data = SentimentData(train, device)
    valid_data = SentimentData(valid, device)
    test_data = SentimentData(test, device)

    assert train_latents.size()[0] == train.shape[0]

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    print("Initial sentiment predictions")
    senti_model.eval()
    predictions, y_test = predict_sentiment(test_loader, senti_model, test_latents)
    if args['dataset'] == 'mosi':
        results = full_loss(predictions, y_test)
    elif args['dataset'] == 'iemocap':
        results = iemocap_loss(predictions, y_test)
    else:
        results = pom_loss(predictions, y_test)

    if model_save_path is not None:
        if 'accuracy' in results:
            with open(os.path.join(model_save_path, 'test_acc_before.txt'), 'w') as f:
                f.write(str(results['accuracy']))
        with open(os.path.join(model_save_path, 'test_results_before.json'), 'w') as f:
            json.dump(results, f, indent=2)

    print("Training sentiment model on sentence embeddings...")
    senti_model.train()
    model_loader = lambda: load_sentiment(model_save_path, embedding_dim, hidden_dim, device)
    train_losses, valid_losses = train_sentiment(args, senti_model, train_loader, train_latents,
            valid_loader, valid_latents, model_loader,
            verbose=verbose, model_save_path=model_save_path)

    if model_save_path is not None:
        with open(os.path.join(model_save_path, 'senti_train_loss.txt'), 'w') as f:
            for loss in train_losses:
                f.write('{}\n'.format(loss))
        with open(os.path.join(model_save_path, 'senti_valid_loss.txt'), 'w') as f:
            for loss in valid_losses:
                f.write('{}\n'.format(loss))

    if not args['early_stopping']:
        if model_save_path is not None:
            save_sentiment(model_save_path, senti_model)
    else:
        print('reloading best')
        model = SentimentModel(embedding_dim, hidden_dim, n_out).to(device)
        checkpoint = torch.load(os.path.join(model_save_path, 'senti.bin'))
        model.load_state_dict(checkpoint['model_state_dict'])

    print("Sentiment predictions after training")
    senti_model.eval()
    predictions, y_test = predict_sentiment(test_loader, senti_model, test_latents)
    if args['dataset'] == 'mosi':
        results = full_loss(predictions, y_test)
    elif args['dataset'] == 'iemocap':
        results = iemocap_loss(predictions, y_test)
    else:
        results = pom_loss(predictions, y_test)

    if model_save_path is not None:
        if 'accuracy' in results:
            with open(os.path.join(model_save_path, 'test_acc_after.txt'), 'w') as f:
                f.write(str(results['accuracy']))
        with open(os.path.join(model_save_path, 'test_results_after.json'), 'w') as f:
            json.dump(results, f, indent=2)

    print("-----------------------------")
