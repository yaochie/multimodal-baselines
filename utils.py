import sys
import json

import h5py
import numpy as np
import data_loader as loader
import torch
from torch.utils.data import Dataset, DataLoader

def load_data(args):
    if args['dataset'] == 'mosi':
        return load_mosi()
    elif args['dataset'] == 'pom':
        return load_pom()
    elif args['dataset'] == 'iemocap':
        return load_iemocap(args)
    else:
        raise ValueError

def load_mosi():
    word2ix = loader.load_word2ix()

    # load glove 300d word embeddings
    word_embeddings = loader.load_word_embedding()
    
    # saved train, valid, test in file so that don't have to
    # regenerate each time
    # Note: loader.load_word_level_features always returns the same split.
    # train, valid, test = loader.load_word_level_features(max_len, tr_proportion)

    train = {}
    valid = {}
    test = {}

    with h5py.File('data/mosi_data.h5', 'r') as f:
        keys = [
            'facet',
            'covarep',
            'text',
            'lengths',
            'label',
            'id',
        ]

        for k in keys:
            train[k] = f['train'][k][:]
            valid[k] = f['valid'][k][:]
            test[k] = f['test'][k][:]

    return word2ix, word_embeddings, (train, valid, test)

def load_pom():
    word2ix = json.load(open('pom/glove_mappings.pom.json', 'r'))
    word_embeddings = np.load('pom/glove.pom.npy')

    # word_embeddings = loader.load_all_glove()

    train = {}
    valid = {}
    test = {}

    with h5py.File('data/pom_data.h5', 'r') as f:
        keys = [
            'facet',
            'covarep',
            'text',
            'label'
        ]

        for k in keys:
            train[k] = f['train'][k][:]
            valid[k] = f['valid'][k][:]
            test[k] = f['test'][k][:]

    print(train['text'].shape)

    # since pom is very long, we might need to
    # truncate so that we don't run out of memory
    MAXLEN = 2000
    # print("truncating ids to", MAXLEN)
    
    x = np.load('pom/pom_train_ids.npy', allow_pickle=False)
    print(x.shape)
    train['text_id'] = x
    x = np.load('pom/pom_valid_ids.npy', allow_pickle=False)
    valid['text_id'] = x
    x = np.load('pom/pom_test_ids.npy', allow_pickle=False)
    test['text_id'] = x

    return word2ix, word_embeddings, (train, valid, test)

def load_iemocap(args):
    word2ix = json.load(open('iemocap/glove_mappings.iemocap.json', 'r'))
    word_embeddings = np.load('iemocap/glove.iemocap.npy')
    
    # word_embeddings = loader.load_all_glove()

    train = {}
    valid = {}
    test = {}

    fname = 'data/iemocap_{}.h5'.format(args['emotion'])

    with h5py.File('data/iemocap_happy.h5', 'r') as f:
        print(list(f['train'].keys()))
        keys = [
            'facet',
            'covarep',
            'text',
            'label'
        ]

        for k in keys:
            train[k] = f['train'][k][:]
            valid[k] = f['valid'][k][:]
            test[k] = f['test'][k][:]

    print(train['text'].shape)

    # save the seq ids into text_ids
    x = np.load('iemocap/iemocap_train_ids.npy', allow_pickle=False)
    train['text_id'] = x
    x = np.load('iemocap/iemocap_valid_ids.npy', allow_pickle=False)
    valid['text_id'] = x
    x = np.load('iemocap/iemocap_test_ids.npy', allow_pickle=False)
    test['text_id'] = x

    return word2ix, word_embeddings, (train, valid, test)

def add_positional_embeddings(args, data):
    """
    PE(pos, 2i)   = sin(pos/10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos/10000^(2i / d_model))
    """

    seq_len = data.shape[1]
    n_points = data.shape[0]

    # do simple loop way
    pos_embed_dim = args['pos_embed_dim']

    idxes = np.arange(seq_len, dtype=np.float32)
    idxes = np.tile(idxes, [n_points, pos_embed_dim, 1])
    idxes = np.transpose(idxes, [0, 2, 1])

    for i in range(pos_embed_dim // 2):
        idxes[2*i,:] = np.sin(idxes[2*i,:] / (10000 ** (2*i / pos_embed_dim)))
        idxes[2*i+1,:] = np.cos(idxes[2*i+1,:] / (10000 ** (2*i / pos_embed_dim)))

    #print(data.shape)
    #print(idxes.shape)

    return np.concatenate([data, idxes], axis=-1)

def normalize_data(train):
    """
    normalize audio and visual features to [-1, 1].
    Also remove any features that are always the same value.

    Also set padding values to -10.
    """
    # normalize audio and visual features to [-1, 1]
    audio_min = train['covarep'].min((0, 1))
    audio_max = train['covarep'].max((0, 1))

    audio_diff = audio_max - audio_min
    audio_nonzeros = audio_diff.nonzero()[0]

    train['covarep'] = train['covarep'][:, :, audio_nonzeros]

    audio_pad = train['covarep'] == 0
    vis_pad = train['facet'] == 0
    audio_mask = (train['covarep'] != 0).astype(int)
    vis_mask = (train['facet'] != 0).astype(int)

    audio_min = train['covarep'].min((0, 1))
    audio_max = train['covarep'].max((0, 1))
    audio_diff = audio_max - audio_min

    padding_idxes = train['covarep']

    vis_min = train['facet'].min((0, 1))
    vis_max = train['facet'].max((0, 1))

    train['covarep'] = (train['covarep'] + audio_min) * 2. / (audio_max - audio_min) - 1.
    train['facet'] = (train['facet'] + vis_min) * 2. / (vis_max - vis_min) - 1.

    train['covarep'][audio_pad] = -10.
    train['facet'][vis_pad] = -10.
    
    return train, {'covarep': audio_mask, 'facet': vis_mask}

class MMData(Dataset):
    def __init__(self, text, audio, visual, masks, text_weights, device):
        super(Dataset, self).__init__()

        if not torch.is_tensor(text):
            text = torch.tensor(text, device=device, dtype=torch.float32)
        if not torch.is_tensor(text_weights):
            text_weights = torch.tensor(text_weights, device=device, dtype=torch.float32)
        if not torch.is_tensor(audio):
            audio = torch.tensor(audio, device=device, dtype=torch.float32)
        if not torch.is_tensor(visual):
            visual = torch.tensor(visual, device=device, dtype=torch.float32)

        torch_masks = {}
        for k in ['text', 'covarep', 'facet']:
            if not torch.is_tensor(masks[k]):
                torch_masks[k] = torch.tensor(masks[k], device=device, dtype=torch.float32)
            else:
                torch_masks[k] = masks[k]

        assert text.size()[0] == audio.size()[0]
        assert audio.size()[0] == visual.size()[0]
        assert text.size()[0] == text_weights.size()[0]

        self.text = text
        self.text_weights = text_weights
        self.audio = audio
        self.visual = visual

        self.text_mask = torch_masks['text']
        self.audio_mask = torch_masks['covarep']
        self.visual_mask = torch_masks['facet']

        self.len = self.text.size()[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (idx, self.text[idx], self.audio[idx], self.visual[idx], self.text_mask[idx],
                self.audio_mask[idx], self.visual_mask[idx], self.text_weights[idx])

class MMDataExtra(MMData):
    def __init__(self, text, audio, visual, masks, text_weights, text_aligned, device):
        super(MMDataExtra, self).__init__(text, audio, visual, masks, text_weights, device)

        if not torch.is_tensor(text_aligned):
            text_aligned = torch.tensor(text_aligned, device=device, dtype=torch.float32)

        self.text_aligned = text_aligned

        if not torch.is_tensor(masks['text_align']):
            ta_mask = torch.tensor(masks['text_align'], device=device, dtype=torch.float32)
        self.text_aligned_mask = ta_mask

    def __getitem__(self, idx):
        return (idx, self.text[idx], self.audio[idx], self.visual[idx], self.text_mask[idx],
                self.audio_mask[idx], self.visual_mask[idx], self.text_weights[idx],
                self.text_aligned[idx], self.text_aligned_mask[idx])

