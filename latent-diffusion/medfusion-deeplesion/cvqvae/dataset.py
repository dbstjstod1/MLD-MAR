## import

import os
import numpy as np
import torch

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
        
        # Normalize to [-1,1]
        input = input/5000
        
        data = {'input': input}

        if self.transform:
            data = self.transform(data)

        return data


##
class ToTensor(object):
    def __call__(self, data):
        
        input = data['input']
  
        input = input.astype(np.float32)

        data = {'input': torch.from_numpy(input)}

        return data
