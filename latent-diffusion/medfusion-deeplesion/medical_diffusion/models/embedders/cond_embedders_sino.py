
import torch.nn as nn
import torch 
from monai.networks.layers.utils import get_act_layer
from medical_diffusion.models.embedders.latent_embedders import VAE, VAEGAN, CVQVAE, VQGAN
#from cvqvae.modules import Model
from cvqvae.modules_sino import Model

class LabelEmbedder(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2):
        super().__init__()
        self.emb_dim = emb_dim
        #self.latent_embedder = CVQVAE.load_from_checkpoint('/home/ysh/Desktop/medfusion-main/scripts/runs/cvqvae/2024_02_25_054855/last.ckpt')
        
        self.latent_embedder_checkpoint = torch.load('/home/ysh/Desktop/CVQ-VAE-main/results/cvqvae_128_2048_4_v2/models/AAPM_cos_closest/best.pt')
        latent_embedder = Model(input_dim=1, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, num_embeddings=2048,embedding_dim=4, distance='cos', perceiver=True)
        latent_embedder.load_state_dict(self.latent_embedder_checkpoint)
        self.latent_embedder = latent_embedder
            
    def forward(self, condition):
        self.latent_embedder.eval()
        with torch.no_grad():
            #c = self.latent_embedder.encode(condition)
            c, _, _, _ = self.latent_embedder.encode(condition)
        return c
