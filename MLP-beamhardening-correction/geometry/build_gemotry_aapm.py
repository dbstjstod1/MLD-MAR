import pdb
import torch
import numpy as np
import functools
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from .AAPMRecon import *
from .AAPMProj import *
import torch.nn.functional as F
from scipy import interpolate
from scipy.ndimage import binary_erosion
from scipy.ndimage import gaussian_filter
import scipy as sp
import scipy.sparse.linalg
from typing import List, Tuple
from PIL import Image
from scipy.signal import convolve2d
import math


device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
water_mua = 0.02 # You must not change this value... it caused from AAPM recon module
sid = 550.
sdd = 950.

nrdetcols = 900
nrcols = 512
nrrows = 512
nrviews = 1000

dalpha = 2. * np.arctan2(1.0 / 2, sdd)
alphas = (np.arange(nrdetcols) - (nrdetcols - 1) / 2 - 1.25) * dalpha

viewangles = np.single(1 * (0 + np.arange(nrviews) / (nrviews) * 2 * np.pi))

def _A_all_cpu(x, FOV):  # forward
    pixsize = FOV / 512
    x0 = 0.0 / pixsize
    y0 = sid / pixsize
    xCor = 0.0 / pixsize
    yCor = 0.0 / pixsize
    xds = np.single(sdd * np.sin(alphas) / pixsize)
    yds = np.single((sid - sdd * np.cos(alphas)) / pixsize)

    x= np.expand_dims(x, axis=0)

    originalImgPtr = np.single(x)
    sinogram = np.zeros([nrviews, nrdetcols, 1], dtype=np.single)
    sinogram = DD2FanProj(nrdetcols, x0, y0, xds, yds, xCor, yCor, viewangles, nrviews, sinogram, nrcols, nrrows,
                          originalImgPtr)
    
    sinogram = sinogram * pixsize  # 1000h 900w 1
    sinogram = np.squeeze(sinogram,axis=2)

    return sinogram

    
def _AINV_all_cpu(sinogram, FOV, ker_type):
    sinogram= np.expand_dims(sinogram, axis=0)
    sinogram = sinogram.transpose(1,0,2)
    cfg = AAPMRecon_init(FOV, ker_type)
    fbp = AAPMRecon_main(sinogram, cfg)  # 512 512 1
    fbp = fbp / 1000 * water_mua + water_mua #to mua
    fbp = np.squeeze(fbp,axis=2)
    return fbp