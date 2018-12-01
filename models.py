import torch
import torch.nn as nn

class AudioVisualGeneratorConcat(nn.Module):
    def __init__(self, audio_embedding_dim, visual_embedding_dim,
                audio_dim, visual_dim, frozen_weights=True):
        super(AudioVisualGeneratorConcat, self).__init__()
        
        self.audio_embedding_dim = audio_embedding_dim
        self.visual_embedding_dim = visual_embedding_dim

        self.embed2audio = nn.ModuleDict({
            'mu': nn.Linear(self.audio_embedding_dim, audio_dim),
            'log_sigma': nn.Linear(self.audio_embedding_dim, audio_dim)
        })

        self.embed2visual = nn.ModuleDict({
            'mu': nn.Linear(self.visual_embedding_dim, visual_dim),
            'log_sigma': nn.Linear(self.visual_embedding_dim, visual_dim)
        })

    def freeze_weights(self):
        # freeze weights
        for layer in self.embed2audio.values():
            for param in layer.parameters():
                param.requires_grad = False

        for layer in self.embed2visual.values():
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, audio_embed, visual_embed):
        audio_mu = self.embed2audio['mu'](audio_embed)
        audio_sigma = self.embed2audio['log_sigma'](audio_embed).exp()

        visual_mu = self.embed2visual['mu'](visual_embed)
        visual_sigma = self.embed2visual['log_sigma'](visual_embed).exp()

        return (audio_mu, audio_sigma), (visual_mu, visual_sigma)

    def init_embeddings(self, word_embeddings):
        n_points = word_embeddings.size()[0]

        aud_embedding = torch.randn(n_points, self.audio_embedding_dim, dtype=torch.float32, device=word_embeddings.device)
        vis_embedding = torch.randn(n_points, self.visual_embedding_dim, dtype=torch.float32, device=word_embeddings.device)
        curr_embedding = torch.cat([word_embeddings, aud_embedding, vis_embedding], dim=1)

        return curr_embedding

class AudioVisualGeneratorMultimodal(nn.Module):
    def __init__(self, embedding_dim, audio_dim, visual_dim, frozen_weights=True):
        super(AudioVisualGeneratorMultimodal, self).__init__()

        self.embedding = None
        self.embedding_dim = embedding_dim

        self.embed2out = nn.ModuleDict({
            'audio': nn.ModuleDict({
                'mu': nn.Linear(self.embedding_dim, audio_dim),
                'log_sigma': nn.Linear(self.embedding_dim, audio_dim)
            }),
            'visual': nn.ModuleDict({
                'mu': nn.Linear(self.embedding_dim, visual_dim),
                'log_sigma': nn.Linear(self.embedding_dim, visual_dim)
            }),
            'audiovisual': nn.ModuleDict({
                'mu': nn.Linear(self.embedding_dim, audio_dim + visual_dim),
                'log_sigma': nn.Linear(self.embedding_dim, audio_dim + visual_dim)
            }),
            'textaudio': nn.ModuleDict({
                'mu': nn.Linear(self.embedding_dim, embedding_dim + audio_dim),
                'log_sigma': nn.Linear(self.embedding_dim, embedding_dim + audio_dim)
            }),
            'textvisual': nn.ModuleDict({
                'mu': nn.Linear(self.embedding_dim, embedding_dim + visual_dim),
                'log_sigma': nn.Linear(self.embedding_dim, embedding_dim + visual_dim)
            }),
            'textaudiovisual': nn.ModuleDict({
                'mu': nn.Linear(self.embedding_dim, embedding_dim + audio_dim + visual_dim),
                'log_sigma': nn.Linear(self.embedding_dim, embedding_dim + audio_dim + visual_dim)
            }),
        })

        if frozen_weights:
            self.freeze_weights()

    def freeze_weights(self):
        # freeze weights
        for module in self.embed2out.values():
            for layer in module.values():
                for param in layer.parameters():
                    param.requires_grad = False

    def init_embedding(self, embedding):
        assert embedding.size()[-1] == self.embedding_dim

        self.embedding = embedding
        self.embedding.requires_grad = True
        self.embedding_dim = self.embedding.size()[-1]

    def forward(self, embeddings):
        to_gen = embeddings

        # from sentence embedding, generate mean and variance of
        # audio and visual features
        # since variance is positive, we exponentiate.
        return {
            out_mods: {
                'mu': module['mu'](to_gen),
                'sigma': module['log_sigma'](to_gen).exp()
            }
            for out_mods, module in self.embed2out.items()
        }

class AudioVisualGenerator(nn.Module):
    def __init__(self, embedding_dim, audio_dim, visual_dim, frozen_weights=True):
        super(AudioVisualGenerator, self).__init__()

        self.embedding = None
        self.embedding_dim = embedding_dim

        self.embed2audio = nn.ModuleDict({
            'mu': nn.Linear(self.embedding_dim, audio_dim),
            'log_sigma': nn.Linear(self.embedding_dim, audio_dim)
        })

        self.embed2visual = nn.ModuleDict({
            'mu': nn.Linear(self.embedding_dim, visual_dim),
            'log_sigma': nn.Linear(self.embedding_dim, visual_dim)
        })

        if frozen_weights:
            self.freeze_weights()

    def freeze_weights(self):
        # freeze weights
        for layer in self.embed2audio.values():
            for param in layer.parameters():
                param.requires_grad = False

        for layer in self.embed2visual.values():
            for param in layer.parameters():
                param.requires_grad = False

    def init_embedding(self, embedding):
        assert embedding.size()[-1] == self.embedding_dim

        self.embedding = embedding
        self.embedding.requires_grad = True
        self.embedding_dim = self.embedding.size()[-1]

    def forward(self, embeddings):
        to_gen = embeddings

        # from sentence embedding, generate mean and variance of
        # audio and visual features
        # since variance is positive, we exponentiate.
        audio_mu = self.embed2audio['mu'](to_gen)
        audio_sigma = self.embed2audio['log_sigma'](to_gen).exp()

        visual_mu = self.embed2visual['mu'](to_gen)
        visual_sigma = self.embed2visual['log_sigma'](to_gen).exp()

        return (audio_mu, audio_sigma), (visual_mu, visual_sigma)
