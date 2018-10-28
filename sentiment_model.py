import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from losses import (get_normal_log_prob, get_word_log_prob_angular,
        get_word_log_prob_dot_prod, full_loss)

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

def train_sentiment(args, model, train_data, train_latents,
            valid_data, valid_latents, model_loader, valid_niter=10,
            verbose=False, model_save_path=None):
    n_epochs = args['n_sentiment_epochs']
    lr = args['sentiment_lr']
    # patience = 5

    n_samples = len(train_data.dataset)
    loss_function = nn.L1Loss(reduce=False)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_losses = []
    valid_losses = []
    best_valid_loss = None
    n_bad = 0
    for i in range(n_epochs):
        epoch_loss = 0
        for j, senti in train_data:
            model.zero_grad()
            senti_predict = model(train_latents[j])
            
            loss = loss_function(senti_predict, senti)
            epoch_loss += loss.sum()

            loss.mean().backward()
            optimizer.step()

        train_losses.append(epoch_loss)
        if i % valid_niter == 0:
            print("Epoch {}: {}".format(i, epoch_loss / n_samples))

            batches = 0
            valid_loss = 0
            with torch.no_grad():
                for j, senti in valid_data:
                    senti_predict = model(valid_latents[j])
                    loss = loss_function(senti_predict, senti)

                    valid_loss += loss.mean()
                    batches += 1
            print("Average validation loss: {}".format(valid_loss / batches))
            valid_losses.append(valid_loss)

            """
            if best_valid_loss is None:
                best_valid_loss = valid_loss
            elif best_valid_loss > valid_loss:
                n_bad = 0
                best_valid_loss = valid_loss
                if model_save_path:
                    save_sentiment(model_save_path, model)
            else:
                # TODO: reload and continue
                n_bad += 1
                if n_bad > patience:
                    print("early stopping...")
                    break
            """

    print("Epoch {}: {}".format(i, epoch_loss / n_samples))
    return train_losses, valid_losses

def train_sentiment_for_latents(args, latents, sentiment_data, device,
            (n_train, n_valid, n_test),
            verbose=False, model_save_path=None, train_idxes=None):
    hidden_dim = args['sentiment_hidden_size']

    embedding_dim = latents.size()[-1]
    senti_model = SentimentModel(embedding_dim, hidden_dim).to(device)

    # prepare data
    train_latents = latents[:n_train]
    valid_latents = latents[n_train:n_train+n_valid]
    test_latents = latents[n_train+n_valid:]

    train, valid, test = sentiment_data
    if train_idxes is not None:
        print(train.shape)
        train = train[train_idxes]
        print(train.shape)

    train_data = SentimentData(train, device)
    valid_data = SentimentData(valid, device)
    test_data = SentimentData(test, device)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    print("Initial sentiment predictions")
    senti_model.eval()
    acc = eval_sentiment(test_loader, senti_model, test_latents)
    if model_save_path is not None:
        with open(os.path.join(model_save_path, 'test_acc_before.txt'), 'w') as f:
            f.write(str(acc))

    print("Training sentiment model on sentence embeddings...")
    senti_model.train()
    model_loader = lambda: load_sentiment(model_save_path, embedding_dim, hidden_dim, device)
    train_losses, valid_losses = train_sentiment(args, senti_model, train_loader, train_latents,
            valid_loader, valid_latents, model_loader,
            verbose=verbose, model_save_path=model_save_path)

    with open(os.path.join(model_save_path, 'senti_train_loss.txt'), 'w') as f:
        for loss in train_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(model_save_path, 'senti_valid_loss.txt'), 'w') as f:
        for loss in valid_losses:
            f.write('{}\n'.format(loss))
    if model_save_path is not None:
        save_sentiment(model_save_path, senti_model)

    print("Sentiment predictions after training")
    senti_model.eval()
    acc = eval_sentiment(test_loader, senti_model, test_latents)
    if model_save_path is not None:
        with open(os.path.join(model_save_path, 'test_acc_after.txt'), 'w') as f:
            f.write(str(acc))
    print("-----------------------------")
