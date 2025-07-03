
import torch.nn as nn
import torch 
from monai.networks.layers.utils import get_act_layer
from medical_diffusion.models.embedders.latent_embedders import VAE, VAEGAN, CVQVAE, VQGAN
from cvqvae.modules_sino import Model_sino
from cvqvae.modules import Model

class LabelEmbedder_deeplesion(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2):
        super().__init__()
        self.emb_dim = emb_dim
        self.latent_embedder_checkpoint = torch.load('/media/mirlab/hdd2/CVQ-VAE-deeplesion/results/Deeplesion_128_4096_4/models/Xma/model_1000.pt')
        latent_embedder = Model(input_dim=1, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, num_embeddings=4096,embedding_dim=4, distance='cos', perceiver=True)
        latent_embedder.load_state_dict(self.latent_embedder_checkpoint)
        self.latent_embedder = latent_embedder
            
    def forward(self, condition):
        self.latent_embedder.eval()
        with torch.no_grad():
            c, _, _, _ = self.latent_embedder.encode(condition)
        return c
    
class LabelEmbedder_deeplesion_sino(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2):
        super().__init__()
        self.emb_dim = emb_dim
        self.latent_embedder_checkpoint = torch.load('/data/1/CVQ-VAE-deeplesion_sino/results/Deeplesion_128_4096_4/models/Sma/model_2000.pt')
        latent_embedder = Model_sino(input_dim=1, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, num_embeddings=4096,embedding_dim=4, distance='cos', perceiver=True)
        latent_embedder.load_state_dict(self.latent_embedder_checkpoint)
        self.latent_embedder = latent_embedder
            
    def forward(self, condition):
        self.latent_embedder.eval()
        with torch.no_grad():
            c, _, _, _ = self.latent_embedder.encode(condition)
        return c

class LabelEmbedder_deeplesion2(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2):
        super().__init__()
        self.emb_dim = emb_dim
        self.latent_embedder_checkpoint = torch.load('/media/mirlab/hdd2/CVQ-VAE-deeplesion/results/Deeplesion_128_4096_4/models/Xma/model_1000.pt')
        latent_embedder = Model(input_dim=1, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, num_embeddings=4096,embedding_dim=4, distance='cos', perceiver=True)
        latent_embedder.load_state_dict(self.latent_embedder_checkpoint)
        self.latent_embedder = latent_embedder
            
    def forward(self, condition):
        self.latent_embedder.eval()
        # import pdb
        # pdb.set_trace()
        with torch.no_grad():
            c1, _, _, _ = self.latent_embedder.encode(condition[:, 0, :, :].unsqueeze(1))
            c2, _, _, _ = self.latent_embedder.encode(condition[:, 1, :, :].unsqueeze(1))
            c = torch.cat([c1,c2],dim=1)
        return c
    
class LabelEmbedder_deeplesion_sino2(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2):
        super().__init__()
        self.emb_dim = emb_dim
        self.latent_embedder_checkpoint = torch.load('/data/1/CVQ-VAE-deeplesion_sino/results/Deeplesion_128_4096_4/models/Sma/model_2000.pt')
        latent_embedder = Model_sino(input_dim=1, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, num_embeddings=4096,embedding_dim=4, distance='cos', perceiver=True)
        latent_embedder.load_state_dict(self.latent_embedder_checkpoint)
        self.latent_embedder = latent_embedder
            
    def forward(self, condition):
        self.latent_embedder.eval()
        # import pdb
        # pdb.set_trace()
        with torch.no_grad():
            c1, _, _, _ = self.latent_embedder.encode(condition[:, 0, :, :].unsqueeze(1))
            c2, _, _, _ = self.latent_embedder.encode(condition[:, 1, :, :].unsqueeze(1))
            c = torch.cat([c1,c2],dim=1)
        return c