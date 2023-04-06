import numpy as np
import torch
from itertools import chain
from torch import nn
from torch.nn import functional as F

class Selection(nn.Module):
    def __init__(self, dim_enc, dim_sprites, group, logger):
        super().__init__()
        self.dim_z = min(dim_enc, dim_sprites)
        logger(f'Selection init with dim_enc={dim_enc}, dim_sprites={dim_sprites} --> dim_z={self.dim_z}')
        self.blank_latent = nn.Parameter(torch.randn((1, self.dim_z)))

        self.linear = nn.Sequential(
            nn.Linear(dim_sprites, self.dim_z),
            nn.LayerNorm(self.dim_z, elementwise_affine=False)
        )

        self.anchors = nn.Sequential(
            nn.Linear(dim_enc, self.dim_z),
            nn.LayerNorm(self.dim_z, elementwise_affine=False)
        )

        self.norm = np.sqrt(self.dim_z)
        self.group, self.log_softmax = group, (self.log_softmax_group if group > 1 else self.log_softmax_)

    def encoder_params(self):
        return self.anchors.parameters()

    def sprite_params(self):
        return chain(self.linear.parameters(), [self.blank_latent])

    def compute_logits(self, x, sprites):
        latents = sprites.masks_.flat_latents()
        latents = self.linear(latents)
        latents = torch.cat([latents, self.blank_latent], dim=0).transpose(1,0)

        B, C, L = x.size()
        x = x.permute(0, 2, 1).reshape(B*L, C)

        a = self.anchors(x)
        return (a @ latents) / self.norm

    def log_softmax_(self, x, logits):
        B, _, L = x.size()
        weights = F.softmax(logits, dim=-1)
        logits = logits.reshape(B, L, -1).permute(1, 0, 2)
        return weights, logits.log_softmax(2)

    def log_softmax_group(self, x, logits):
        B, _, L = x.size()
        weights = F.softmax(logits, dim=-1)
        logits = weights.reshape(B, L, -1).permute(1, 0, 2)
        K = logits.size()[-1]
        all_logits, blank_logits = torch.split(logits, [K-1, 1], dim=-1)
        all_logits = all_logits.view(L, B, (K-1)//self.group, self.group).sum(-1)
        logits = torch.cat([all_logits, blank_logits], dim=-1)
        return weights, logits.log()

    def forward(self, x, sprites):
        """Predicts probabilities for each sprite at each position."""
        B, _, L = x.size()
        logits = self.compute_logits(x, sprites)
        weights, log_probs = self.log_softmax(x, logits)

        output = {'w': weights, 'logits': logits, 'log_probs': log_probs}
        if not self.training:
            # only one sprite per position
            weights = torch.eye(weights.shape[-1]).to(weights)[weights.argmax(-1)] 
            output['selection'] = weights.reshape(B, L, -1).permute(2, 0, 1)

        # prototypes and masks concatenated
        sprite = torch.cat([
                torch.cat([sprites.prototypes, torch.zeros_like(sprites.prototypes[0]).unsqueeze(0)], dim=0),
                torch.cat([sprites.masks, torch.zeros_like(sprites.masks[0]).unsqueeze(0)], dim=0),
            ], dim=1)

        # multiplication of probabilities after the softmax and sprites (masks+colors)
        S = (weights[..., None, None, None] * sprite[None, ...]).sum(1)
        _, C, H, W = S.size()       
        output['S'] = S.reshape(B, L, 4, H, W).permute(1, 0, 2, 3, 4)

        return output

    def compute_sprites(self,sprites, selection):
        """Computes sprite at each position from selection in path-select mode."""
        K, B, n_cells = selection.size()
        weights = selection.permute(1,2,0).reshape(B*n_cells, K) #one-hot encoded selection

        # prototypes and masks concatenated
        sprite = torch.cat([
                torch.cat([sprites.prototypes, torch.zeros_like(sprites.prototypes[0]).unsqueeze(0)], dim=0),
                torch.cat([sprites.masks, torch.zeros_like(sprites.masks[0]).unsqueeze(0)], dim=0),
            ], dim=1)  

        # multiplication of probabilities after the softmax and sprites (masks+colors)
        S = (weights[..., None, None, None] * sprite[None, ...]).sum(1)
        _, C, H, W = S.size()    

        return S.reshape(B, n_cells, 4, H, W).permute(1, 0, 2, 3, 4)
