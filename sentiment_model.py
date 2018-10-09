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
    print(y_test.size())
    full_loss(predictions, y_test)

def train_sentiment(args, data, model, latents, valid_niter=10):
    n_epochs = args['n_sentiment_epochs']
    lr = args['sentiment_lr']

    n_samples = len(data.dataset)
    loss_function = nn.L1Loss(reduce=False)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for i in range(n_epochs):
        epoch_loss = 0
        for j, senti in data:
            model.zero_grad()
            senti_predict = model(latents[j])
            
            loss = loss_function(senti_predict, senti)
            epoch_loss += loss.sum()

            loss.mean().backward()
            optimizer.step()

        if i % valid_niter == 0:
            print("Epoch {}: {}".format(i, epoch_loss / n_samples))

def train_sentiment_for_latents(args, latents, sentiment_data, device):
    hidden_dim = args['sentiment_hidden_size']

    embedding_dim = latents.size()[-1]
    senti_model = SentimentModel(embedding_dim, hidden_dim).to(device)

    print("Initial sentiment predictions")
    eval_sentiment(sentiment_data, senti_model, latents)

    print("Training sentiment model on sentence embeddings...")
    train_sentiment(args, sentiment_data, senti_model, latents)

    print("Sentiment predictions after training")
    eval_sentiment(sentiment_data, senti_model, latents)
    print("-----------------------------")
