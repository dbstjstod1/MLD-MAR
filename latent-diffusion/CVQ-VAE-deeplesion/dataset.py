## import

import os
import numpy as np
import torch
import torch.utils.data as udata
from numpy.random import RandomState
import random
import h5py
import pdb
## DATA LOADER
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, nu, nv, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.nu = nu
        self.nv = nv

        self.data_dir_input = os.path.join(self.data_dir, 'input')
        
        lst_input = os.listdir(self.data_dir_input)
        lst_input.sort()
        
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_input)

    def __getitem__(self, index):
        
        input = np.fromfile(os.path.join(self.data_dir_input, self.lst_input[index]), dtype=np.float32).reshape([1, self.nv, self.nu])
        
        # Normalize to [-1,1] # Turn off Normalize if mua domain training
        #input = -1 + 2*(input-(-1500))/((5000)-(-1500)) #no need to be clip -1 1 for embedder
        
        # Normalize to [-1,1] # mua domain training
        input = -1 + 2*(input)/(0.15) #no need to be clip -1 1 for embedder
        
        data = {'input': input}

        if self.transform:
            data = self.transform(data)

        return data
    
def image_get_minmax():
    return 0.0, 1.0

def proj_get_minmax():
    return 0.0, 4.0   
 
def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    # data = data * 255.0
    data = data * 2. - 1.
    data = data.astype(np.float32)
    #data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    data = np.transpose(np.expand_dims(data, 2), (2, 0, 1))
    return data

class MARTrainDataset_train(udata.Dataset):
    def __init__(self, dir):
        super().__init__()
        self.dir = dir
        self.patch_size = 416
        self.txtdir = os.path.join(self.dir, 'train_640geo_dir.txt')
        self.mat_files = open(self.txtdir, 'r').readlines()
        self.file_num = len(self.mat_files)
        self.rand_state = RandomState(66)
    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        gt_dir = self.mat_files[idx]
        random_mask = random.randint(0, 89)  # include 89
        #random_mask = random.randint(0, 9)  # for demo
        file_dir = gt_dir[:-6]
        data_file = file_dir + str(random_mask) + '.h5'
        abs_dir = os.path.join(self.dir, 'train_640geo/', data_file)
        gt_absdir = os.path.join(self.dir,'train_640geo/', gt_dir[:-1])
        gt_file = h5py.File(gt_absdir, 'r')
        gt_file.close()
        file = h5py.File(abs_dir, 'r')
        Xma = file['ma_CT'][()]
        Sma = file['ma_sinogram'][()]
        XLI =file['LI_CT'][()]
        file.close()
        Xma = normalize(Xma, image_get_minmax())
        Xma = torch.Tensor(Xma)
        Sma = normalize(Sma, proj_get_minmax())
        Sma = torch.Tensor(Sma)
        XLI = normalize(XLI, image_get_minmax())
        XLI = torch.Tensor(XLI)
        
        #data = {'input': XLI}
        data = {'input': Xma}
        #pdb.set_trace()
        
        #if self.transform:
        #    data = self.transform(data)
            
        return data
    
class MARTrainDataset_test(udata.Dataset):
    def __init__(self, dir):
        super().__init__()
        self.dir = dir
        self.patch_size = 416
        self.txtdir = os.path.join(self.dir, 'test_640geo_dir.txt')
        self.mat_files = open(self.txtdir, 'r').readlines()
        self.file_num = len(self.mat_files)
        self.rand_state = RandomState(66)
    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        gt_dir = self.mat_files[idx]
        #random_mask = random.randint(0, 89)  # include 89
        random_mask = random.randint(0, 9)  # for demo
        file_dir = gt_dir[:-6]
        data_file = file_dir + str(random_mask) + '.h5'
        abs_dir = os.path.join(self.dir, 'test_640geo/', data_file)
        gt_absdir = os.path.join(self.dir,'test_640geo/', gt_dir[:-1])
        gt_file = h5py.File(gt_absdir, 'r')
        gt_file.close()
        file = h5py.File(abs_dir, 'r')    
        Xma = file['ma_CT'][()]
        XLI =file['LI_CT'][()]
        Sma = file['ma_sinogram'][()]
        file.close()
        Xma = normalize(Xma, image_get_minmax())
        Xma = torch.Tensor(Xma)
        Sma = normalize(Sma, proj_get_minmax())
        Sma = torch.Tensor(Sma)
        XLI = normalize(XLI, image_get_minmax())
        XLI = torch.Tensor(XLI)
        
        #data = {'input': XLI}
        data = {'input': Xma}
        
        #if self.transform:
        #    data = self.transform(data)
            
        return data
    
class MARTrainDataset_test2(udata.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.dir = data_dir
        self.patch_size = 416
        
        self.data_dir_Xma = os.path.join(self.dir, 'Xma')
        self.data_dir_Xgt = os.path.join(self.dir, 'Xgt')
        self.data_dir_XLI = os.path.join(self.dir, 'XLI')
        
        lst_Xma = os.listdir(self.data_dir_Xma)
        lst_Xgt = os.listdir(self.data_dir_Xgt)
        lst_XLI = os.listdir(self.data_dir_XLI)
        
        lst_Xma.sort()
        lst_Xgt.sort()
        lst_XLI.sort()
        
        self.lst_Xma = lst_Xma
        self.lst_Xgt = lst_Xgt
        self.lst_XLI = lst_XLI
        
    def __len__(self):
        return len(self.lst_Xma)

    def __getitem__(self, index):
        #print(self.lst_Xma[index])

        Xma = np.fromfile(os.path.join(self.data_dir_Xma, self.lst_Xma[index]), dtype=np.float32).reshape([self.patch_size, self.patch_size])
        Xgt = np.fromfile(os.path.join(self.data_dir_Xgt, self.lst_Xgt[index]), dtype=np.float32).reshape([self.patch_size, self.patch_size])
        XLI = np.fromfile(os.path.join(self.data_dir_Xgt, self.lst_Xgt[index]), dtype=np.float32).reshape([self.patch_size, self.patch_size])
        
        Xma = normalize(Xma, image_get_minmax())
        Xgt = normalize(Xgt, image_get_minmax())
        XLI = normalize(XLI, image_get_minmax())
        
        Xma = torch.Tensor(Xma)
        Xgt = torch.Tensor(Xgt)
        XLI = torch.Tensor(XLI)
        
        # plt.imshow(Xma[0,:,:], interpolation='nearest')
        # plt.show()
        # plt.imshow(Xgt[0,:,:], interpolation='nearest')
        # plt.show()
        # plt.imshow(XLI[0,:,:], interpolation='nearest')
        # plt.show()
        
        data = {'input': Xma}

        return data
##
class ToTensor(object):
    def __call__(self, data):
        
        input = data['input']
  
        input = input.astype(np.float32)

        data = {'input': torch.from_numpy(input)}

        return data
