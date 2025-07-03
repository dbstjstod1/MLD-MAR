import torch
import torch.nn as nn
import torch.nn.functional as F

from quantise import VectorQuantiser
from diffusionmodules.model import Encoder, Decoder
from diffusionmodules.perceivers import LPIPS

class Model(nn.Module):
    def __init__(self, ddconfig, n_embed, embed_dim, commitment_cost=0.25, distance='l2', 
                 anchor='closest', first_batch=False, contras_loss=True, perceiver=None, perceptual_loss_weight = 1.0):
        super(Model, self).__init__()
        
        self._encoder = Encoder(**ddconfig)
        self._pre_vq_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        
        self._vq_vae = VectorQuantiser(n_embed, embed_dim, commitment_cost, distance=distance, 
                                       anchor=anchor, first_batch=first_batch, contras_loss=contras_loss)
        
        self._post_vq_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self._decoder = Decoder(**ddconfig)
        
        self.perceiver = LPIPS().eval() if perceiver is True else None 
        self.perceptual_loss_weight = perceptual_loss_weight

    def encode(self, x):
        z_e_x = self._encoder(x)
        z_e_x = self._pre_vq_conv(z_e_x)
        quantized, loss, (perplexity, encodings, _) = self._vq_vae(z_e_x)
        return quantized, loss, perplexity, encodings
    
    def decode(self, quantized):
        quantized = self._post_vq_conv(quantized)
        x_recon = self._decoder(quantized)
        return x_recon
    
    def perception_loss(self, pred, target):
        if (self.perceiver is not None):
            self.perceiver.eval()
            return self.perceiver(pred, target)*self.perceptual_loss_weight
        else:
            return 0 

    def forward(self, x):
        quantized, loss, perplexity, encodings = self.encode(x)
        x_recon = self.decode(quantized)
        p_loss = torch.mean(self.perception_loss(x_recon, x)) if self.perceiver is not None else 0 
        return x_recon, loss, p_loss, perplexity, encodings