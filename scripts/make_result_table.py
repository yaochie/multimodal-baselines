"""
Combine results with config info
"""

import os
import sys
from collections import defaultdict
import json
import csv

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# collect: accuracy before and after finetuning, before and after training
# sentiment model
folder = 'first_model_2'
cfolder = 'first_model_len_20'
#folder = cfolder = 'weighted_loss'
#folder = 'model_saves/weighted_loss'

# N x 4 x 3 (# of configs x # accs x # runs)
accs = defaultdict(dict)
data = {
    'config_num': [],
    'run_num': [],
    'pre_before': [],
    'pre_after': [],
    'post_before': [],
    'post_after': []
}
configs = []
for d in os.listdir(os.path.join('model_saves', folder)):
    d2 = os.path.join('model_saves', folder, d)

    stuff = d.split('_')
    config_num = int(stuff[1])
    configs.append(config_num)
    run_num = int(stuff[3])

    f1 = os.path.join(d2, 'pre', 'acc_before.txt')
    f2 = os.path.join(d2, 'pre', 'acc_after.txt')
    f3 = os.path.join(d2, 'post', 'acc_before.txt')
    f4 = os.path.join(d2, 'post', 'acc_after.txt')

    if not (os.path.isfile(f1) and os.path.isfile(f2) and os.path.isfile(f3) and os.path.isfile(f4)):
        continue

    pre_acc_before = float(open(f1, 'r').read())
    pre_acc_after = float(open(f2, 'r').read())
    post_acc_before = float(open(f3, 'r').read())
    post_acc_after = float(open(f4, 'r').read())

    data['config_num'].append(config_num)
    data['run_num'].append(run_num)
    data['pre_before'].append(pre_acc_before)
    data['pre_after'].append(pre_acc_after)
    data['post_before'].append(post_acc_before)
    data['post_after'].append(post_acc_after)

    accs[config_num][run_num] = {
            'pre_before': pre_acc_before,
            'pre_after': pre_acc_after,
            'post_before': post_acc_before,
            'post_after': post_acc_after
    }

#df = pd.DataFrame(data=data)
#print(df.columns)
#print(df.dtypes)

#g = sns.catplot(x='config_num', y='post_after', data=df)
#plt.show()
#sys.exit()

"""
Final plots:
    one plot of accuracy at all 4 times, with error bars
    get top 10 configs, plot their accs at different times with error bars
"""
pre_acc_before = [x['pre_before'] for c in accs.values() for x in c.values()]
pre_acc_after = [x['pre_after'] for c in accs.values() for x in c.values()]
post_acc_before = [x['post_before'] for c in accs.values() for x in c.values()]
post_acc_after = [x['post_after'] for c in accs.values() for x in c.values()]

all_stats = np.stack([pre_acc_before, pre_acc_after, post_acc_before, post_acc_after])
print(all_stats.shape)

accs2 = []
for c, v in accs.items():
    pre_before = []
    pre_after = []
    post_before = []
    post_after = []
    for x in v.values():
        pre_before.append(x['pre_before'])
        pre_after.append(x['pre_after'])
        post_before.append(x['post_before'])
        post_after.append(x['post_after'])

    accs2.append((
        c,
        np.max(pre_before),
        np.max(pre_after),
        np.max(post_before),
        np.max(post_after)
    ))
print(len(accs2))

accs_dict = {
    c[0]: {'max_pre_before': c[1], 'max_pre_after': c[2], 'max_post_before': c[3], 'max_post_after': c[4]}
    for c in accs2
}
print(len(accs_dict))

combined_info = []
for c_num in sorted(list(set(configs))):
    config_file = os.path.join('configs', cfolder, 'config_{}.json'.format(c_num))
    conf = json.load(open(config_file, 'r'))
    if conf['config_num'] in accs_dict:
        conf.update(accs_dict[conf['config_num']])
    combined_info.append(conf)

print(len(combined_info))

with open('results/{}.csv'.format(folder), 'w') as f:
    writer = csv.DictWriter(f, fieldnames=combined_info[0].keys())
    writer.writeheader()
    for c in combined_info:
        writer.writerow(c)
