"""
Plot accuracy on sentiment analysis before and after finetuning word embeddings
"""

import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# collect: accuracy before and after finetuning, before and after training
# sentiment model
#folder = 'model_saves/first_model_2'
folder = 'model_saves/weighted_loss'

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
for d in os.listdir(folder):
    d2 = os.path.join(folder, d)

    stuff = d.split('_')
    config_num = int(stuff[1])
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

# plt.bar(range(4), np.mean(all_stats, axis=1), yerr=np.std(all_stats, axis=1))
#plt.bar(range(4), np.max(all_stats, axis=1))
#plt.show()

# accs2 = []
# for c, v in accs.items():
#     pre_before = []
#     pre_after = []
#     post_before = []
#     post_after = []
#     for x in v.values():
#         pre_before.append(x['pre_before'])
#         pre_after.append(x['pre_after'])
#         post_before.append(x['post_before'])
#         post_after.append(x['post_after'])
# 
#     accs2.append((
#         c,
#         (np.mean(pre_before), np.std(pre_before)),
#         (np.mean(pre_after), np.std(pre_after)),
#         (np.mean(post_before), np.std(post_before)),
#         (np.mean(post_after), np.std(post_after))
#     ))
# 
# 
# sorted_accs = sorted(accs2, key=lambda x: x[-1][0], reverse=True)
# 
# #xes = [str(x[0]) for x in sorted_accs[:10]]
# post_after = [x[4][0] for x in sorted_accs[:10]]
# post_after_err = [x[4][1] for x in sorted_accs[:10]]
# plt.errorbar(range(10), post_after, yerr=post_after_err)
# pre_after = [x[2][0] for x in sorted_accs[:10]]
# pre_after_err = [x[2][1] for x in sorted_accs[:10]]
# plt.errorbar(range(10), pre_after, yerr=pre_after_err)
# 
# plt.show()

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


sorted_accs = sorted(accs2, key=lambda x: x[-1], reverse=True)
sorted_accs = sorted_accs[:10]
print(sorted_accs)

# make into dataframe
data = {
    'config': [],
    'type': [],
    'error': [],
    #'pre_before': [x[1] for x in sorted_accs],
    #'pre_after': [x[2] for x in sorted_accs],
    #'post_before': [x[3] for x in sorted_accs],
    #'post_after': [x[4] for x in sorted_accs]
}

# data['config'].extend([x[0] for x in sorted_accs])
# data['error'].extend([x[1] for x in sorted_accs])
# data['type'].extend(['pre_before' for _ in range(len(sorted_accs))])
data['config'].extend([x[0] for x in sorted_accs])
data['error'].extend([x[2] for x in sorted_accs])
data['type'].extend(['pre_after' for _ in range(len(sorted_accs))])
# data['config'].extend([x[0] for x in sorted_accs])
# data['error'].extend([x[3] for x in sorted_accs])
# data['type'].extend(['post_before' for _ in range(len(sorted_accs))])
data['config'].extend([x[0] for x in sorted_accs])
data['error'].extend([x[4] for x in sorted_accs])
data['type'].extend(['post_after' for _ in range(len(sorted_accs))])

df = pd.DataFrame(data)
print(df)

#configs = df.loc[df['type'] == 'post_after']

g = sns.catplot(x='config', y='error', hue='type', data=df, kind='bar')
g.set_ylabels('Sentiment analysis accuracy')

#xes = [str(x[0]) for x in sorted_accs[:10]]
#post_after = [x[4] for x in sorted_accs[:10]]
#plt.plot(range(10), post_after)
#pre_after = [x[2] for x in sorted_accs[:10]]
#plt.plot(range(10), pre_after)

plt.show()
