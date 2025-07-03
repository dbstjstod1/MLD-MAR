
from pathlib import Path
import torch 
from torchvision import utils 
import math 
from medical_diffusion.models.pipelines import DiffusionPipeline
from medical_diffusion.data.datasets import test_image2, test_image_clinic
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
# from skimage import io
import os
import pdb
from medical_diffusion.models.embedders.latent_embedders import CVQVAE, VQGAN
import numpy as np
import struct

if __name__ == "__main__":
    path_in = Path.cwd()/'results/deeplesion_mlp_step1/Xin'
    path_in.mkdir(parents=True, exist_ok=True)
    
    path_out = Path.cwd()/'results/deeplesion_mlp_step1/Xout'
    path_out.mkdir(parents=True, exist_ok=True)
    
    path_ground_truth = Path.cwd()/'results/deeplesion_mlp_step1/Xgt'
    path_ground_truth.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    device = torch.device('cuda')

    # ------------ Load Model ------------
    pipeline = DiffusionPipeline.load_from_checkpoint('/home/mirlab/Desktop/AAPM_Challanges/deeplesion_checkpoint/diffusion_deeplesion_mlp/2025_04_02_121300/epoch=56-step=7100.ckpt')
    # pipeline = DiffusionPipeline.load_from_checkpoint('/home/mirlab/Desktop/AAPM_Challanges/deeplesion_checkpoint/diffusion_deeplesion_Xma/2024_11_17_162421/epoch=63-step=8000.ckpt')
    pipeline.to(device)
 
    # --------- Generate Samples  -------------------
    steps = 1
    use_ddim = True
    images = {}
    n_samples = 1
    
    data_dir='/media/mirlab/hdd2/DeepLesion_metal/test_raw'
    inner_dir = 'test_640geo/'

    sampler_kwargs = {
        'un_cond': None,
    }
    
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
    for imag_idx in range(200): # 200 for all test data
        print(imag_idx)
        for mask_idx in tqdm(range(10)): # 10
            Xnet, Xgt = test_image2(data_dir, imag_idx, mask_idx)
            # Xnet = test_image_clinic(data_dir, imag_idx, mask_idx)
            Xnet = Xnet.unsqueeze(dim=0)

            X = pipeline.sample(num_samples=n_samples, img_size=(4,104,104), condition=Xnet, **sampler_kwargs, steps=steps, use_ddim=use_ddim).detach()

            X_cpu = X.data.cpu().numpy().squeeze()
            X_cpu = (X_cpu+1)/2
            
            f = open(os.path.join(path_out, '%03d_%03d.raw'%(imag_idx, mask_idx)), "wb")
            output_p = np.reshape(X_cpu.squeeze(), 416*416)
            myfmt = 'f' * len(output_p)
            bin = struct.pack(myfmt, *output_p)
            f.write(bin)
            f.close
            
            # XLI_cpu = XLI.data.cpu().numpy().squeeze()
            # XLI_cpu = (XLI_cpu+1)/2
            # XLI_cpu = (XLI_cpu-0.192)/0.192*1000
            
            # f = open(os.path.join(path_in, '%03d_%03d.raw'%(imag_idx, mask_idx)), "wb")
            # output_p = np.reshape(XLI_cpu.squeeze(), 416*416)
            # myfmt = 'f' * len(output_p)
            # bin = struct.pack(myfmt, *output_p)
            # f.write(bin)
            # f.close
            
            # Xma_cpu = Xma.data.cpu().numpy().squeeze()
            # Xma_cpu = (Xma_cpu+1)/2
            # Xma_cpu = (Xma_cpu-0.192)/0.192*1000
            
            # f = open(os.path.join(path_in2, '%03d_%03d.raw'%(imag_idx, mask_idx)), "wb")
            # output_p = np.reshape(Xma_cpu.squeeze(), 416*416)
            # myfmt = 'f' * len(output_p)
            # bin = struct.pack(myfmt, *output_p)
            # f.write(bin)
            # f.close
            
            # Xgt_cpu = Xgt.data.cpu().numpy().squeeze()
            # Xgt_cpu = (Xgt_cpu+1)/2
            # Xgt_cpu = (Xgt_cpu-0.192)/0.192*1000
            
            # f = open(os.path.join(path_ground_truth, '%03d_%03d.raw'%(imag_idx, mask_idx)), "wb")
            # output_p = np.reshape(Xgt_cpu.squeeze(), 416*416)
            # myfmt = 'f' * len(output_p)
            # bin = struct.pack(myfmt, *output_p)
            # f.write(bin)
            # f.close

        
