import json
import itertools
import os
import csv
import random

#folder = 'frozen_weights'
#name = 'frozen_weights'
dir_path = os.path.dirname(os.path.realpath(__file__))
folder = name = 'multimodal_search2'
folder = os.path.join(dir_path, folder)

if not os.path.isdir(folder):
    os.mkdir(folder)

params = {
    'sentiment_hidden_size': [100, 150],
    'lr': [1e-3, 1e-4],
    'sentiment_lr': [1e-1, 1e-2],
    'seq_len': [20],
    #'word_sim_metric': ['angular', 'dot_prod'],
    'word_sim_metric': ['angular'],
    'n_epochs': [100, 200],
    'freeze_weights': [False],
    'n_sentiment_epochs': [400],
    'word_loss_weight': [0.001, 0.002],
    'likelihood_weight': [0.0001, 0.001],
    'pos_embed_dim': [2, 4],
    'e2e': [True],
    'norm': ['layer_norm', 'batch_norm'],
    'optimizer': ['sgd', 'adam'],
}

param_keys = []
param_values = []
for k, v in params.items():
    param_keys.append(k)
    param_values.append(v)

print(len(list(itertools.product(*param_values))))

csvfile = open(os.path.join(dir_path, '{}.csv'.format(name)), 'w')
writer = csv.DictWriter(csvfile, fieldnames=list(params.keys()) + ['config_num'])
writer.writeheader()

configs = []
for x in itertools.product(*param_values):
    config = {
        k: v for k, v in zip(param_keys, x)
    }
    configs.append(config)

random.shuffle(configs)
for i, config in enumerate(configs):
    config['config_num'] = i
    with open(os.path.join(folder, 'config_{}.json'.format(i)), 'w') as f:
        json.dump(config, f)

    writer.writerow(config)

csvfile.close()
