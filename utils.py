import h5py
import numpy as np
import data_loader as loader
import torch
from torch.utils.data import Dataset, DataLoader

def load_data():
    word2ix = loader.load_word2ix()

    # load glove 300d word embeddings
    word_embeddings = loader.load_word_embedding()
    
    # saved train, valid, test in file so that don't have to
    # regenerate each time
    # Note: loader.load_word_level_features always returns the same split.
    # train, valid, test = loader.load_word_level_features(max_len, tr_proportion)

    with h5py.File('mosi_data.h5', 'r') as f:
        keys = [
            'facet',
            'covarep',
            'text',
            'lengths',
            'label',
            'id',
        ]
        train = {}
        valid = {}
        test = {}

        for k in keys:
            train[k] = f['train'][k][:]
            valid[k] = f['valid'][k][:]
            test[k] = f['test'][k][:]

    return word2ix, word_embeddings, (train, valid, test)

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
    def __init__(self, text, audio, visual, masks, device):
        super(Dataset, self).__init__()
        
        if not torch.is_tensor(text):
            text = torch.tensor(text, device=device, dtype=torch.long)
        if not torch.is_tensor(audio):
            audio = torch.tensor(audio, device=device, dtype=torch.float32)
        if not torch.is_tensor(visual):
            visual = torch.tensor(visual, device=device, dtype=torch.float32)

        for k in ['text', 'covarep', 'facet']:
            if not torch.is_tensor(masks[k]):
                masks[k] = torch.tensor(masks[k], device=device, dtype=torch.float32)

        assert text.size()[0] == audio.size()[0]
        assert audio.size()[0] == visual.size()[0]

        self.text = text
        self.audio = audio
        self.visual = visual

        self.text_mask = masks['text']
        self.audio_mask = masks['covarep']
        self.visual_mask = masks['facet']

        self.len = self.text.size()[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (idx, self.text[idx], self.audio[idx], self.visual[idx], self.text_mask[idx],
                self.audio_mask[idx], self.visual_mask[idx])
