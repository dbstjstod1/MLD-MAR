import os
import os.path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py
from PIL import Image
import pdb
import torch.utils.data as udata
from scipy.interpolate import interp1d
from .build_gemotry import initialization, imaging_geo
#from .build_gemotry_aapm import _A_all_cpu, _AINV_all_cpu

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
    #data = data.astype(np.float32)
    #data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    return data

param = initialization()
ray_trafo, FBPOper = imaging_geo(param)

class MARTrainDataset(udata.Dataset):
    def __init__(self, dir, patchSize):
        super().__init__()
        self.dir = dir
        self.patch_size = patchSize
        
        self.data_dir_Xma = os.path.join(self.dir, 'Xma')
        self.data_dir_Xgt = os.path.join(self.dir, 'Xgt')
        
        lst_Xma = os.listdir(self.data_dir_Xma)
        lst_Xgt = os.listdir(self.data_dir_Xgt)
        
        lst_Xma.sort()
        lst_Xgt.sort()
        
        self.lst_Xma = lst_Xma
        self.lst_Xgt = lst_Xgt
    def __len__(self):
        return len(self.lst_Xma)

    def __getitem__(self, index):
        #print(self.lst_Xma[index])
        water_mua = 0.192
        mask_thre = 2500 /1000 * water_mua + water_mua 

        Xma = np.fromfile(os.path.join(self.data_dir_Xma, self.lst_Xma[index]), dtype=np.float32).reshape([1, self.patch_size, self.patch_size])
        Xgt = np.fromfile(os.path.join(self.data_dir_Xgt, self.lst_Xgt[index]), dtype=np.float32).reshape([1, self.patch_size, self.patch_size])
        
        Xma = Xma/1000*water_mua+water_mua #HU to mua
        Xgt = Xgt/1000*water_mua+water_mua
        
        # Xma = np.array(Image.fromarray(Xma[0,:,:]).resize((416, 416), Image.Resampling.BILINEAR))
        # Xgt = np.array(Image.fromarray(Xgt[0,:,:]).resize((416, 416), Image.Resampling.BILINEAR))
        
        # Xma = np.expand_dims(Xma, 0)
        # Xgt = np.expand_dims(Xgt, 0)
        
        Sgt = np.asarray(ray_trafo(Xgt[0,:,:]))
        Sgt = np.expand_dims(Sgt,0)

        M = np.zeros_like(Xma)
        [rowindex, colindex] = np.where(Xma[0,:,:] > mask_thre)
        M[0, rowindex, colindex] = 1
        Pmetal_kev = np.asarray(ray_trafo(M[0,:,:]))
        Tr = np.zeros_like(Sgt)
        Tr[0,:,:] = Pmetal_kev > 0
        
        SLI = np.zeros_like(Sgt)
        SLI_pad = np.zeros_like(np.pad(SLI,(3,3)))
        SLI_pad = interpolate_projection(np.pad(Sgt[0,:,:],(3,3),mode='edge'), np.pad(Tr[0,:,:],(3,3)))
        SLI[0,:,:] = SLI_pad[3:-3,3:-3]
        
        Sma = np.copy(SLI)

        XLI = np.zeros_like(Xma)
        XLI[0,:,:] = np.asarray(FBPOper(SLI[0,:,:]))
        
        
        Xma = normalize(Xma, image_get_minmax())
        Xgt = normalize(Xgt, image_get_minmax())
        XLI = normalize(XLI, image_get_minmax())
        Sma = normalize(Sma, proj_get_minmax())
        Sgt = normalize(Sgt, proj_get_minmax())
        SLI = normalize(SLI, proj_get_minmax())
        Tr = 1 -Tr.astype(np.float32)
        
        # plt.imshow(Xma[0,:,:], interpolation='nearest')
        # plt.show()
        # plt.imshow(Xgt[0,:,:], interpolation='nearest')
        # plt.show()
        # plt.imshow(XLI[0,:,:], interpolation='nearest')
        # plt.show()
        # plt.imshow(Xprior[0,:,:], interpolation='nearest')
        # plt.show()
        # plt.imshow(Sma[0,:,:], interpolation='nearest')
        # plt.show()
        # plt.imshow(Sgt[0,:,:], interpolation='nearest')
        # plt.show()
        # plt.imshow(SLI[0,:,:], interpolation='nearest')
        # plt.show()
        # plt.imshow(Tr[0,:,:], interpolation='nearest')
        # plt.show()
        # pdb.set_trace()

        return torch.from_numpy(Xma), torch.from_numpy(XLI), torch.from_numpy(Xgt), torch.from_numpy(M), \
               torch.from_numpy(Sma), torch.from_numpy(SLI), torch.from_numpy(Sgt), torch.from_numpy(Tr)
               
def interpolate_projection(proj, metalTrace):
    # projection linear interpolation
    # Input:
    # proj:         uncorrected projection
    # metalTrace:   metal trace in projection domain (binary image)
    # Output:
    # Pinterp:      linear interpolation corrected projection
    Pinterp = proj.copy()
    for i in range(Pinterp.shape[0]):
        mslice = metalTrace[i]
        pslice = Pinterp[i]

        metalpos = np.nonzero(mslice==1)[0]
        nonmetalpos = np.nonzero(mslice==0)[0]
        pnonmetal = pslice[nonmetalpos]
        pslice[metalpos] = interp1d(nonmetalpos,pnonmetal)(metalpos)
        Pinterp[i] = pslice

    return Pinterp

