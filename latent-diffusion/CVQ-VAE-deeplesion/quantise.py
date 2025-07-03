import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange

class DiagonalGaussianDistribution(nn.Module):

    def forward(self, x):
        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        sample = torch.randn(mean.shape, generator=None, device=x.device)
        z = mean + std * sample

        batch_size = x.shape[0]
        var = torch.exp(logvar)
        kl = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar)/batch_size

        return z, kl 

class VectorQuantizer(nn.Module):
    def __init__(self, num_embed, embed_dim, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embed
        self.emb_channels = embed_dim
        self.beta = beta
  
        self.embedder = nn.Embedding(num_embed, embed_dim)
        self.embedder.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        assert z.shape[1] == self.emb_channels, "Channels of z and codebook don't match"
        z_ch = torch.moveaxis(z, 1, -1) # [B, C, *] -> [B, *, C]
        z_flattened = z_ch.reshape(-1, self.emb_channels) # [B, *, C] -> [Bx*, C], Note: or use contiguous() and view()

        # distances from z to embeddings e: (z - e)^2 = z^2 + e^2 - 2 e * z
        dist = (    torch.sum(z_flattened**2, dim=1, keepdim=True) 
                 +  torch.sum(self.embedder.weight**2, dim=1)
                -2* torch.einsum("bd,dn->bn", z_flattened, self.embedder.weight.t())
        ) # [Bx*, num_embeddings]

        min_encoding_indices = torch.argmin(dist, dim=1) # [Bx*]
        z_q = self.embedder(min_encoding_indices) # [Bx*, C]
        z_q = z_q.view(z_ch.shape) # [Bx*, C] -> [B, *, C] 
        z_q = torch.moveaxis(z_q, -1, 1) # [B, *, C] -> [B, C, *]
 
        # Compute Embedding Loss 
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
     
        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, loss

class VectorQuantiser(nn.Module):
    """
    Improved version over vector quantiser, with the dynamic initialisation
    for these unoptimised "dead" points.
    num_embed: number of codebook entry
    embed_dim: dimensionality of codebook entry
    beta: weight for the commitment loss
    distance: distance for looking up the closest code
    anchor: anchor sampled methods
    first_batch: if true, the offline version of our model
    contras_loss: if true, use the contras_loss to further improve the performance
    """
    def __init__(self, num_embed, embed_dim, beta, distance='cos', 
                 anchor='probrandom', first_batch=False, contras_loss=False):
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False

        self.pool = FeaturePool(self.num_embed, self.embed_dim)
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))

    
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.embed_dim)

        # clculate the distance
        if self.distance == 'l2':
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(self.embedding.weight ** 2, dim=1) + \
                2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(self.embedding.weight, 'n d-> d n'))
        elif self.distance == 'cos':
            # cosine distances from z to embeddings e_j 
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
            normed_codebook = F.normalize(self.embedding.weight, dim=1)
            d = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))

        # encoding
        sort_distance, indices = d.sort(dim=1)
        # look up the closest point for the indices
        encoding_indices = indices[:,-1]
        encodings = torch.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # quantise and unflatten
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        # count
        #import pdb
        #pdb.set_trace()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        min_encodings = encodings

        # online clustered reinitialisation for unoptimized points
        if self.training:
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                # closest sampling
                if self.anchor == 'closest':
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1,:]]
                # feature pool based random sampling
                elif self.anchor == 'random':
                    random_feat = self.pool.query(z_flattened.detach())
                # probabilitical based random sampling
                elif self.anchor == 'probrandom':
                    norm_distance = F.softmax(d.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = torch.exp(-(self.embed_prob*self.num_embed*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.embed_dim)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True
            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0)/self.num_embed)):,:].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0)*1/2),:]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                loss +=  contra_loss

        return z_q, loss, (perplexity, min_encodings, encoding_indices)

class FeaturePool():
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """
    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1)/ pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)    
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size: # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features

        return self.features