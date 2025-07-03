import sys


import os
import argparse

import torch
import torch as th
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import yaml
import skimage
import math

from models.hash_encoder import *
# from models.mepnet import EquiCNN
#from models.UNet_openai import create_model

import pdb
import matplotlib.pyplot as plt
import yaml
from scipy.io import loadmat
import numpy as np

from scipy.ndimage import gaussian_filter
# from patch_diffusion import dist_util
# from patch_diffusion.script_util import (
#     NUM_CLASSES,
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     add_dict_to_argparser,
#     args_to_dict,
#     add_dict_to_dict,
#     dict_to_dict
# )
# from geometry.build_gemotry import initialization, build_gemotry
# from geometry.syndeeplesion_data import test_image
import imageio.v2 as imageio
from tqdm import tqdm
from torch_radon.radon import FanBeam
from torchvision.transforms import Resize
from sklearn.cluster import k_means
from scipy import interpolate
import scipy.io as sio
import scipy
from kornia.filters import gaussian_blur2d
import re
from typing import List, Tuple
import torch.nn.init as init
import torch.nn as nn

from odl.contrib import torch as odl_torch
from geometry.build_gemotry import initialization, build_gemotry
import time

def image_get_minmax():
    return 0.0, 1.0

def proj_get_minmax():
    return 0.0, 4.0

def normalize(data, minmax):
    data_min, data_max = minmax
    # data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    # data = data * 255.0
    data = data * 2. - 1.
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    return data

def extract_numbers(input_string):
    numbers = re.findall(r'\d+', input_string)
    return numbers

def query_and_save(epoch):
    global GT_GRID
    
    recon_vol = np.zeros((256, 256, 256), dtype=np.uint8)
    for j in range(256):
        grid_j = GT_GRID[:, j, :, :]  # with psf [256,256,3]
        grid_j = grid_j.reshape(-1, 3)
        grid_j = grid_j.cuda()
        
        recon_output_j = model(grid_j)  # [256, 256, intensity]
        recon_output_j = recon_output_j.reshape(256, 256, 1)
        
        recon_output_j = th.clip(recon_output_j, -1.0, 1.0)
        recon_output_j = recon_output_j.permute(2, 0, 1)[None, :]
        
        recon_output_j = (recon_output_j + 1) / 2
        
        recon_vol[j, :, :] = np.round((recon_output_j).detach().cpu().numpy().squeeze() * 255).astype(np.uint8)  #
    
    skimage.io.imsave(f'./results/epoch{epoch}.tif', recon_vol)


def clear(x):
    x = x.detach().cpu().squeeze().numpy()
    return x


def neighbours(i: int, j: int, max_i: int, max_j: int) -> List[Tuple[int, int]]:
    """
    Returns 4-connected neighbours for given pixel point.
    :param i: i-th index position
    :param j: j-th index position
    :param max_i: max possible i-th index position
    :param max_j: max possible j-th index position
    """
    pairs = []
    
    for n in [-1, 1]:
        if 0 <= i + n <= max_i:
            pairs.append((i + n, j))
        if 0 <= j + n <= max_j:
            pairs.append((i, j + n))
    
    return pairs

def poisson_blend_fast(img_s, mask, img_t, device):
    """
    Consistent Poisson blending with optimized performance.
    """
    img_s = img_s.detach().cpu().squeeze().numpy()
    mask = mask.detach().cpu().squeeze().numpy()
    img_t = img_t.detach().cpu().squeeze().numpy()
    img_s_h, img_s_w = img_s.shape
    
    # Identify non-zero mask indices
    ys, xs = np.where(mask > 0)
    nnz = len(ys)
    
    # Map masked pixels to linear indices
    im2var = -np.ones(mask.shape, dtype=int)
    im2var[ys, xs] = np.arange(nnz)
    
    # Prepare sparse matrix data
    row_indices = []
    col_indices = []
    values = []
    b = np.zeros(4 * nnz)  # Larger b to account for all equations
    
    e = 0
    for n in range(nnz):
        y, x = ys[n], xs[n]
        
        for n_y, n_x in neighbours(y, x, img_s_h - 1, img_s_w - 1):
            # Diagonal entry
            row_indices.append(e)
            col_indices.append(im2var[y, x])
            values.append(1)  # Current pixel weight
            b[e] = img_s[y, x] - img_s[n_y, n_x]
            
            # Neighbor contributions
            if im2var[n_y, n_x] != -1:  # Internal neighbor
                row_indices.append(e)
                col_indices.append(im2var[n_y, n_x])
                values.append(-1)
            else:  # Boundary neighbor
                b[e] += img_t[n_y, n_x]
            e += 1
    
    # Construct sparse matrix A
    A = scipy.sparse.coo_matrix((values, (row_indices, col_indices)), shape=(e, nnz))
    
    # Solve linear system A * v = b
    v = scipy.sparse.linalg.lsqr(A, b[:e])[0]
    
    # Map solution back to output image
    img_t_out = img_t.copy()
    for n in range(nnz):
        y, x = ys[n], xs[n]
        img_t_out[y, x] = v[int(im2var[y, x])]
    
    img_t_out = img_t_out.astype(np.float32)
    img_t_out = th.from_numpy(img_t_out)
    h = img_t_out.shape[0]
    w = img_t_out.shape[1]
    img_t_out = img_t_out.reshape(1, 1, h, w)
    
    return img_t_out.to(device)


def init_weights(m):
    if isinstance(m, nn.Linear):  # nn.Linear에 대해서만 초기화 적용
        # torch.nn.init.xavier_uniform_(m.weight)
        init.uniform_(m.weight, a=-((6 / 64) ** 0.5), b=((6 / 64) ** 0.5))
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

data_path = './test_data'
data_path = os.path.join(data_path, 'Sma')
data_Num = len(os.listdir((data_path)))
measurement_lst = os.listdir((data_path)) # Sma
# SLDM_lst = os.listdir((data_path.replace('Sma', 'Sino_LDM')))
measurement_lst.sort()
# SLDM_lst.sort()

# dist_util.setup_dist()

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print('Device:', device)
print('Current cuda device:', th.cuda.current_device())
print('Count of using GPUs:', th.cuda.device_count())
cudnn.benchmark = True

# recommend hash_encoder for much faster training instead of SIREN or MLPs.
model_type ='hash_encoder'
param = initialization()

num_proj = 640

angles = np.linspace(-np.pi/2*1, np.pi/2*3, 640)
radon = FanBeam(641, angles, 1075, 1075, param.param['su']/param.reso/641)

max_epoch = 100

# sds_weight = 0.1
# resize1 = Resize(416)
# resize2 = Resize(512)
# fp = lambda x: radon.forward(x) * param.reso

para_ini = initialization()
fp,fbp = build_gemotry(para_ini)
op_modfp = odl_torch.OperatorModule(fp)
op_modpT = odl_torch.OperatorModule(fbp)

# mask = th.zeros([1, 1, 640, 641]).cuda()
# mask[:, :, ::sparsity, :] = 1
image_x = 6
image_y = 1

if model_type == 'hash_encoder':
    grid_y_, grid_x_ = th.meshgrid([
        th.linspace(0, 1, steps=image_y),
        th.linspace(0, 1, steps=image_x)])
    GT_GRID = th.stack([grid_y_, grid_x_], dim=-1)
elif model_type == 'Equi':
    grid_y_, grid_x_ = th.meshgrid([
        th.linspace(0, 1, steps=416),
        th.linspace(0, 1, steps=416)])
    GT_GRID = th.stack([grid_y_, grid_x_], dim=-1)
else:
    grid_y_, grid_x_ = th.meshgrid([
        th.linspace(-1, 1, steps=416),
        th.linspace(-1, 1, steps=416)])
    GT_GRID = th.stack([grid_y_, grid_x_], dim=-1)


# loss_fn = th.nn.L1Loss()
loss_fn = th.nn.MSELoss()
##
iteration=5000

# for i in range(1,2): #data_Num
for i in range(2000): #data_Num
#for i in reversed(range(data_Num)): #data_Num
    
    if model_type == 'hash_encoder':
        model_INR = MLP_hash(n_inputs=2, output_dim=1, n_levels=32, n_features_per_level=1, log2_hashmap_size=22,
                             base_resolution=16, per_level_scale=8 ** (1 / 31)).cuda()
        model_INR.apply(init_weights)
        
    model_INR.cuda()
    
    model_INR.train()
    optim = th.optim.Adam(model_INR.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=0)  # 0.003
    
    XLi = np.fromfile(os.path.join(data_path.replace('Sma', 'XLI'), measurement_lst[i]), dtype=np.float32).reshape(
        [416, 416])
    MetalMask = np.fromfile(os.path.join(data_path.replace('Sma', 'mask'), measurement_lst[i]),
                            dtype=np.float32).reshape([416, 416])
    known = np.fromfile(os.path.join(data_path, measurement_lst[i]), dtype=np.float32).reshape([640, 641])

    Tr = np.fromfile(os.path.join(data_path.replace('Sma', 'Tr'), measurement_lst[i]), dtype=np.float32).reshape(
        [640, 641])
    Xgt = np.fromfile(os.path.join(data_path.replace('Sma', 'Xgt'), measurement_lst[i]), dtype=np.float32).reshape(
        [416, 416])
    SLI = np.fromfile(os.path.join(data_path.replace('Sma', 'SLI'), measurement_lst[i]), dtype=np.float32).reshape(
        [640, 641])
    
    Xgt = th.from_numpy(Xgt).reshape(1, 1, 416, 416)
    Xgt = Xgt.cuda()
    
    MetalMask = th.from_numpy(MetalMask).reshape(1, 1, 416, 416)
    MetalMask = MetalMask.cuda()

    known = th.from_numpy(known).reshape(1, 1, 640, 641)
    known = 1*known.cuda()
    known = torch.clamp(known, max=3.197) #Starvation option

    SLI = th.from_numpy(SLI).reshape(1, 1, 640, 641)
    SLI = 1*SLI.cuda()
    
    Xma = op_modpT(known)*num_proj*2
    # clear((Xma)).tofile(
    #     os.path.join('./results/NAF/%d_results'%num_proj, measurement_lst[i])
    # )
    # pdb.set_trace()
    
    # clear((op_modfp(Xma))).tofile(
    #     os.path.join('./results/NAF/%d_results'%num_proj, measurement_lst[i])
    # )
    # pdb.set_trace()
    
    Tr = th.from_numpy(Tr).reshape(1, 1, 640, 641)
    Tr = Tr.cuda()
    
    XLi = th.from_numpy(XLi).reshape(1,1,416, 416)
    XLi = XLi.cuda()

    grid = GT_GRID.reshape(2,1,image_y,image_x).cuda()
    grid = grid.reshape(-1, 2)  # x,y
    metal_proj = op_modfp(MetalMask)
    # clear((metal_proj)).tofile(
    #     os.path.join('./results/NAF/%d_results/metal_proj'%num_proj, measurement_lst[i])
    # )
    # pdb.set_trace()
    eps = 1e-12
    start_time = time.time()
    # torch.cuda.reset_peak_memory_stats()
    for iter in range(iteration):
        model_INR.train()

        optim.zero_grad()
        if model_type == 'hash_encoder':
            output = model_INR(grid)
            mu = output[:,0]
            powers = list(range(5, -1, -1))      # [10, 9, ..., 0]
            
            # Assume metal_proj shape: [1, 1, H, W] and same device as mu
            proj_poly = sum([mu[i] * metal_proj ** p for i, p in enumerate(powers)])
            mu_bh = proj_poly #proj_domain

        dc_loss = loss_fn(known - mu_bh, SLI)
        
        train_loss = dc_loss
        
        print(f'iter: {iter}, dc_loss: {dc_loss.item()}')

        train_loss.backward(retain_graph=True)
        optim.step()
        # allocated = torch.cuda.memory_allocated() / 1024**2  # in MB
        # cached = torch.cuda.memory_reserved() / 1024**2      # for older PyTorch: memory_cached
        # peak = torch.cuda.max_memory_allocated() / 1024**2
        # print(f"Current allocated: {allocated:.2f} MB")
        # print(f"Current reserved (cached): {cached:.2f} MB")
        # print(f"Peak allocated: {peak:.2f} MB")
        # pdb.set_trace()
        
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")      
    # pdb.set_trace()
    # Save final result
    with th.no_grad():

        # model_INR.eval()
        # clear(op_modpT(mu_bh)*num_proj*2).tofile(
        #     os.path.join('./results/NAF/%d_results/bhc'%num_proj, 'bhc_' + measurement_lst[i])
        # )
        # clear(known-mu_bh).tofile(
        #     os.path.join('./results/NAF/%d_results/corr_sino'%num_proj, 'correct_sino_' + measurement_lst[i])
        # )
        # clear(op_modpT(known-mu_bh)*num_proj*2).tofile(
        #     os.path.join('./results/NAF/%d_results/corr_image'%num_proj, 'correct_img_' + measurement_lst[i])
        # )
        # mu_np = mu.detach().cpu().numpy()
        # filename = measurement_lst[i].removesuffix('.raw')
        # np.save(os.path.join('./results/NAF/%d_results/mlp_param'%num_proj,filename + ".npy"), mu_np)
        print(f'{i}th image is saved')
