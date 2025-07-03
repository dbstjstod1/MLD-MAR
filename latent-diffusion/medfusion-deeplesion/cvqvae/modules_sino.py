import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantise import VectorQuantiser
#from diffusionmodules.perceivers import LPIPS
from diffusionmodules.lpips import LPIPS

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1),
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, output_channels):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = up_conv(num_hiddens,num_hiddens//2)
        #self._conv_trans_1 = nn.ConvTranspose2d(num_hiddens,num_hiddens//2,kernel_size=4,stride=2,padding=1)
        
        self._conv_trans_2 = up_conv(num_hiddens//2,output_channels)
        #self._conv_trans_2 = nn.ConvTranspose2d(num_hiddens//2,output_channels,kernel_size=4,stride=2,padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)
    

class Model_sino(nn.Module):
    def __init__(self, input_dim, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost=0.25, distance='l2', 
                 anchor='closest', first_batch=False, contras_loss=True, perceiver=None, perceptual_loss_weight=1.0):
        super(Model_sino, self).__init__()
        
        self._encoder = Encoder(input_dim, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        
        self._vq_vae = VectorQuantiser(num_embeddings, embedding_dim, commitment_cost, distance=distance, 
                                       anchor=anchor, first_batch=first_batch, contras_loss=contras_loss)
        
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens,
                                input_dim)
        
        self.perceiver = LPIPS().eval() if perceiver is True else None 
        self.perceptual_loss_weight = perceptual_loss_weight

    def encode(self, x):
        z_e_x = self._encoder(x)
        z_e_x = self._pre_vq_conv(z_e_x)
        quantized, loss, (perplexity, encodings, _) = self._vq_vae(z_e_x)
        return quantized, loss, perplexity, encodings
    
    def decode(self, quantized):
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
    
