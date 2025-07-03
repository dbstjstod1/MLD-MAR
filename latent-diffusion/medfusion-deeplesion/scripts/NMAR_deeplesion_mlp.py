
from pathlib import Path
import torch 
from torchvision import utils 
import math 
from medical_diffusion.models.pipelines import DiffusionPipeline
from medical_diffusion.data.datasets import test_image_NMAR_clinic
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
# from skimage import io
import os
import pdb
from medical_diffusion.models.embedders.latent_embedders import CVQVAE, VQGAN
import numpy as np
import struct
from odl.contrib import torch as odl_torch
from build_gemotry import initialization, build_gemotry
para_ini = initialization()
fp,bp = build_gemotry(para_ini)

if __name__ == "__main__":
    path_ground_truth = Path.cwd()/'results/deeplesion_mlp_clinic/Xnmar'
    path_ground_truth.mkdir(parents=True, exist_ok=True)
    
    path_ground_truth2 = Path.cwd()/'results/deeplesion_mlp_clinic/M'
    path_ground_truth2.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    device = torch.device('cuda')

    data_dir='/media/mirlab/hdd2/DeepLesion_metal/test'
    inner_dir = 'test_640geo/'

    fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
    for imag_idx in range(1): # 200 for all test data
        print(imag_idx)
        for mask_idx in tqdm(range(10)): # 10
            S_NMAR, M = test_image_NMAR_clinic(data_dir, imag_idx, mask_idx, inner_dir)
            S_NMAR = S_NMAR.data.cpu().numpy().squeeze()
            
            X_recon_cpu = bp(S_NMAR)
            X_recon_cpu = np.array(X_recon_cpu)
            f = open(os.path.join(path_ground_truth, '%03d_%03d.raw'%(imag_idx, mask_idx)), "wb")
            output_p = np.reshape(X_recon_cpu.squeeze(), 416*416)
            myfmt = 'f' * len(output_p)
            bin = struct.pack(myfmt, *output_p)
            f.write(bin)
            f.close
            
            # f = open(os.path.join(path_ground_truth2, '%03d_%03d.raw'%(imag_idx, mask_idx)), "wb")
            # output_p = np.reshape(M.squeeze(), 416*416)
            # myfmt = 'f' * len(output_p)
            # bin = struct.pack(myfmt, *output_p)
            # f.write(bin)
            # f.close
 

        
