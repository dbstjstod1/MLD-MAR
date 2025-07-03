import os
import torch.utils.data as data 
import torch 
from torch import nn
from pathlib import Path 
from torchvision import transforms as T
import pandas as pd 
import numpy as np
from PIL import Image
import cv2
from scipy.interpolate import interp1d
from medical_diffusion.data.augmentation.augmentations_2d import Normalize, ToTensor16bit

class SH_Dataset_latent_emb(data.Dataset):
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
        
        input = np.fromfile(os.path.join(self.data_dir_input, self.lst_input[index]), dtype=np.float32).reshape([self.nv, self.nu])

        # Make in range [-1.0 1.0] with float32 type
        input = -1 + 2*(input-(-1500))/((5000)-(-1500)) #no need to be clip -1 1 for embedeer
        
        # Add channel for tensor transform
        input = np.expand_dims(input,axis=0)
        
        data = {'source': input, 'target': input}       
        
        if self.transform:
            data = self.transform(data)

        return data
    
class SH_Dataset_sino(torch.utils.data.Dataset):
    def __init__(self, data_dir, nu, nv, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.nu = nu
        self.nv = nv
        #self.nz = nz
        #self.nlam = nlam
        self.data_dir_input = os.path.join(self.data_dir, 'input')
        self.data_dir_label = os.path.join(self.data_dir, 'label')
        self.data_dir_feature = os.path.join(self.data_dir, 'prior_sino')
        self.data_dir_feature2 = os.path.join(self.data_dir, 'inverse_mask')        
        
        lst_input = os.listdir(self.data_dir_input)
        lst_label = os.listdir(self.data_dir_label)
        lst_feature = os.listdir(self.data_dir_feature)
        lst_feature2 = os.listdir(self.data_dir_feature2)
        
        lst_input.sort()
        lst_label.sort()
        lst_feature.sort()
        lst_feature2.sort()
        
        self.lst_input = lst_input
        self.lst_label = lst_label
        self.lst_feature = lst_feature
        self.lst_feature2 = lst_feature2
        
    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        
        input = np.fromfile(os.path.join(self.data_dir_input, self.lst_input[index]), dtype=np.float32).reshape([1, self.nv, self.nu])
        label = np.fromfile(os.path.join(self.data_dir_label, self.lst_label[index]), dtype=np.float32).reshape([1, self.nv, self.nu])
        prior = np.fromfile(os.path.join(self.data_dir_feature, self.lst_feature[index]), dtype=np.float32).reshape([1, self.nv, self.nu])
        mask = np.fromfile(os.path.join(self.data_dir_feature2, self.lst_feature2[index]), dtype=np.float32).reshape([1, self.nv, self.nu])
        
        #Data Standarization
        mean_input = np.mean(input*mask)
        std_input = np.std(input*mask)
        max_input = np.max(input*mask)
        min_input = np.min(input*mask)
        
        mean_prior = np.mean(prior*mask)
        std_prior = np.std(prior*mask)
        max_prior = np.max(prior*mask)
        min_prior = np.min(prior*mask)
        
        norm_prior = (max_input-min_input)/(max_prior-min_prior)*(prior-min_prior)+min_input
        stand_prior = std_input/std_prior*(norm_prior-mean_prior)+mean_input
        
        #Norm for NMAR
        norm_input = (input+1)/(stand_prior+1.0)*0.5
        norm_label = (label+1)/(stand_prior+1.0)*0.5
        
        #Masking in the metal trace
        norm_input = mask*norm_input

        data = {'source': norm_label, 'target': norm_input}      

        if self.transform:
            data = self.transform(data)

        return data
    
class SH_Dataset_dentium(data.Dataset):
    def __init__(self, data_dir, nu, nv, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.nu = nu
        self.nv = nv
        #self.nz = nz
        #self.nlam = nlam
        self.data_dir_input = os.path.join(self.data_dir, 'input')
        self.data_dir_label = os.path.join(self.data_dir, 'label')
        
        lst_input = os.listdir(self.data_dir_input)
        lst_label = os.listdir(self.data_dir_label)
        
        lst_input.sort()
        lst_label.sort()
        
        self.lst_input = lst_input
        self.lst_label = lst_label

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        
        input = np.fromfile(os.path.join(self.data_dir_input, self.lst_input[index]), dtype=np.float32).reshape([self.nv, self.nu])
        label = np.fromfile(os.path.join(self.data_dir_label, self.lst_label[index]), dtype=np.float32).reshape([self.nv, self.nu])
        
        # Normalize to [-1,1] # mua domain training
        #input = 2*input
        #label = 2*label
        
        input = -1 + 2*(input)/(0.15) #no need to be clip -1 1 for embedder
        label = -1 + 2*(label)/(0.15)
        
        # Add channel for tensor transform
        input = np.expand_dims(input,axis=0)
        label = np.expand_dims(label,axis=0)
        
        data = {'source': label, 'target': input}         
        
        if self.transform:
            data = self.transform(data)

        return data
    
class SH_Dataset(data.Dataset):
    def __init__(self, data_dir, nu, nv, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.nu = nu
        self.nv = nv
        #self.nz = nz
        #self.nlam = nlam
        self.data_dir_input = os.path.join(self.data_dir, 'input')
        self.data_dir_label = os.path.join(self.data_dir, 'label')
        
        lst_input = os.listdir(self.data_dir_input)
        lst_label = os.listdir(self.data_dir_label)
        
        lst_input.sort()
        lst_label.sort()
        
        self.lst_input = lst_input
        self.lst_label = lst_label

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        
        input = np.fromfile(os.path.join(self.data_dir_input, self.lst_input[index]), dtype=np.float32).reshape([self.nv, self.nu])
        label = np.fromfile(os.path.join(self.data_dir_label, self.lst_label[index]), dtype=np.float32).reshape([self.nv, self.nu])
        
        # Outlier clip
        label = np.clip(label,-1500,5000)
        #input = np.clip(input,-1500,5000)
        
        # Make in range [-1.0 1.0] with float32 type
        input = -1 + 2*(input-(-1500))/((5000)-(-1500)) 
        label = -1 + 2*(label-(-1500))/((5000)-(-1500))
        
        # Normalize to [-1,1] # mua domain training
        #input = -1 + 2*(input)/(0.15) #no need to be clip -1 1 for embedder
        #label = -1 + 2*(label)/(0.15)
        
        # Add channel for tensor transform
        input = np.expand_dims(input,axis=0)
        label = np.expand_dims(label,axis=0)
        
        data = {'source': label, 'target': input}         
        
        if self.transform:
            data = self.transform(data)

        return data
    
class SH_Dataset_finetune(data.Dataset):
    def __init__(self, data_dir, nu, nv, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.nu = nu
        self.nv = nv
        #self.nz = nz
        #self.nlam = nlam
        self.data_dir_input = os.path.join(self.data_dir, 'LDM_step5_nmar_recon_withmask_label')
        self.data_dir_label = os.path.join(self.data_dir, 'label')
        
        lst_input = os.listdir(self.data_dir_input)
        lst_label = os.listdir(self.data_dir_label)
        
        lst_input.sort()
        lst_label.sort()
        
        self.lst_input = lst_input
        self.lst_label = lst_label

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        
        input = np.fromfile(os.path.join(self.data_dir_input, self.lst_input[index]), dtype=np.float32).reshape([self.nv, self.nu])
        label = np.fromfile(os.path.join(self.data_dir_label, self.lst_label[index]), dtype=np.float32).reshape([self.nv, self.nu])
        
        # Outlier clip
        label = np.clip(label,-1500,5000)
        
        # Make in range [-1.0 1.0] with float32 type
        input = -1 + 2*(input-(-1500))/((5000)-(-1500)) 
        label = -1 + 2*(label-(-1500))/((5000)-(-1500))
        
        # Normalize to [-1,1] # mua domain training
        #input = -1 + 2*(input)/(0.15) #no need to be clip -1 1 for embedder
        #label = -1 + 2*(label)/(0.15)
        
        # Add channel for tensor transform
        input = np.expand_dims(input,axis=0)
        label = np.expand_dims(label,axis=0)
        
        data = {'source': label, 'target': input}         
        
        if self.transform:
            data = self.transform(data)

        return data
    
class SimpleDataset2D(data.Dataset):
    def __init__(
        self,
        path_root,
        item_pointers =[],
        crawler_ext = 'tif', # other options are ['jpg', 'jpeg', 'png', 'tiff'],
        transform = None,
        image_resize = None,
        augment_horizontal_flip = False,
        augment_vertical_flip = False, 
        image_crop = None,
    ):
        super().__init__()
        self.path_root = Path(path_root)
        self.crawler_ext = crawler_ext
        if len(item_pointers):
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.run_item_crawler(self.path_root, self.crawler_ext) 

        if transform is None: 
            self.transform = T.Compose([
                T.Resize(image_resize) if image_resize is not None else nn.Identity(),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
                T.CenterCrop(image_crop) if image_crop is not None else nn.Identity(),
                T.ToTensor(),
                # T.Lambda(lambda x: torch.cat([x]*3) if x.shape[0]==1 else x),
                # ToTensor16bit(),
                # Normalize(), # [0, 1.0]
                # T.ConvertImageDtype(torch.float),
                T.Normalize(mean=0.5, std=0.5) # WARNING: mean and std are not the target values but rather the values to subtract and divide by: [0, 1] -> [0-0.5, 1-0.5]/0.5 -> [-1, 1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.item_pointers)

    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root/rel_path_item
        # img = Image.open(path_item) 
        img = self.load_item(path_item)
        return {'uid':rel_path_item.stem, 'source': self.transform(img)}
    
    def load_item(self, path_item):
        return Image.open(path_item).convert('RGB') 
        # return cv2.imread(str(path_item), cv2.IMREAD_UNCHANGED) # NOTE: Only CV2 supports 16bit RGB images 
    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        return [path.relative_to(path_root) for path in Path(path_root).rglob(f'*.{extension}')]

    def get_weights(self):
        """Return list of class-weights for WeightedSampling"""
        return None 


class AIROGSDataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = pd.read_csv(self.path_root.parent/'train_labels.csv', index_col='challenge_id')
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        uid = self.labels.index[index]
        path_item = self.path_root/f'{uid}.jpg'
        img = self.load_item(path_item)
        str_2_int = {'NRG':0, 'RG':1} # RG = 3270, NRG = 98172 
        target = str_2_int[self.labels.loc[uid, 'class']]
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        return {'source': self.transform(img), 'target':target}
    
    def get_weights(self):
        n_samples = len(self)
        weight_per_class = 1/self.labels['class'].value_counts(normalize=True) # {'NRG': 1.03, 'RG': 31.02}
        weights = [0] * n_samples
        for index in range(n_samples):
            target = self.labels.iloc[index]['class']
            weights[index] = weight_per_class[target]
        return weights
    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []

class MSIvsMSS_Dataset(SimpleDataset2D):
    # https://doi.org/10.5281/zenodo.2530835
    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root/rel_path_item
        img = self.load_item(path_item)
        uid = rel_path_item.stem
        str_2_int = {'MSIMUT':0, 'MSS':1}
        target = str_2_int[path_item.parent.name] #
        return {'uid':uid, 'source': self.transform(img), 'target':target}


class MSIvsMSS_2_Dataset(SimpleDataset2D):
    # https://doi.org/10.5281/zenodo.3832231
    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root/rel_path_item
        img = self.load_item(path_item)
        uid = rel_path_item.stem
        str_2_int = {'MSIH':0, 'nonMSIH':1} # patients with MSI-H = MSIH; patients with MSI-L and MSS = NonMSIH)
        target = str_2_int[path_item.parent.name] 
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        return {'source': self.transform(img), 'target':target}


class CheXpert_Dataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode = self.path_root.name
        labels = pd.read_csv(self.path_root.parent/f'{mode}.csv', index_col='Path')
        self.labels = labels.loc[labels['Frontal/Lateral'] == 'Frontal'].copy()
        self.labels.index = self.labels.index.str[20:]
        self.labels.loc[self.labels['Sex'] == 'Unknown', 'Sex'] = 'Female' # Affects 1 case, must be "female" to match stats in publication
        self.labels.fillna(2, inplace=True) # TODO: Find better solution, 
        str_2_int = {'Sex': {'Male':0, 'Female':1}, 'Frontal/Lateral':{'Frontal':0, 'Lateral':1}, 'AP/PA':{'AP':0, 'PA':1}}
        self.labels.replace(str_2_int, inplace=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        rel_path_item = self.labels.index[index]
        path_item = self.path_root/rel_path_item
        img = self.load_item(path_item)
        uid = str(rel_path_item)
        target = torch.tensor(self.labels.loc[uid, 'Cardiomegaly']+1, dtype=torch.long)  # Note Labels are -1=uncertain, 0=negative, 1=positive, NA=not reported -> Map to [0, 2], NA=3
        return {'uid':uid, 'source': self.transform(img), 'target':target}

    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []

class CheXpert_2_Dataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        labels = pd.read_csv(self.path_root/'labels/cheXPert_label.csv', index_col=['Path', 'Image Index']) # Note: 1 and -1 (uncertain) cases count as positives (1), 0 and NA count as negatives (0)
        labels = labels.loc[labels['fold']=='train'].copy() 
        labels = labels.drop(labels='fold', axis=1)

        labels2 = pd.read_csv(self.path_root/'labels/train.csv', index_col='Path')
        labels2 = labels2.loc[labels2['Frontal/Lateral'] == 'Frontal'].copy()
        labels2 = labels2[['Cardiomegaly',]].copy()
        labels2[ (labels2 <0) | labels2.isna()] = 2 # 0 = Negative, 1 = Positive, 2 = Uncertain
        labels = labels.join(labels2['Cardiomegaly'], on=["Path",], rsuffix='_true')
        # labels = labels[labels['Cardiomegaly_true']!=2]

        self.labels = labels 
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path_index, image_index = self.labels.index[index]
        path_item = self.path_root/'data'/f'{image_index:06}.png'
        img = self.load_item(path_item)
        uid = image_index
        target = int(self.labels.loc[(path_index, image_index), 'Cardiomegaly'])
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        return {'source': self.transform(img), 'target':target}
    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []
    
    def get_weights(self):
        n_samples = len(self)
        weight_per_class = 1/self.labels['Cardiomegaly'].value_counts(normalize=True)
        # weight_per_class = {2.0: 1.2, 1.0: 8.2, 0.0: 24.3}
        weights = [0] * n_samples
        for index in range(n_samples):
            target = self.labels.loc[self.labels.index[index], 'Cardiomegaly']
            weights[index] = weight_per_class[target]
        return weights
    
class ToTensor(object):
    def __call__(self, data):
        
        label, input = data['target'], data['source']

        label = label.astype(np.float32)        
        input = input.astype(np.float32)

        data = {'source': torch.from_numpy(input), 'target': torch.from_numpy(label)}

        return data

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
    #data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    data = np.transpose(np.expand_dims(data, 2), (2, 0, 1))
    return data

import torch.utils.data as udata
from numpy.random import RandomState
import random
import h5py
from .build_gemotry import initialization, build_gemotry
import PIL
from PIL import Image
import pdb
from odl.contrib import torch as odl_torch
param = initialization()
ray_trafo, FBPOper = build_gemotry(param)
op_modfp = odl_torch.OperatorModule(ray_trafo)
op_modpT = odl_torch.OperatorModule(FBPOper)

def clear(x):
    x = x.detach().cpu().squeeze().numpy()
    return x

class MARTrainDataset_train_mlp(udata.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.dir = data_dir
        self.patch_size = 416
        self.txtdir = os.path.join(self.dir, 'train_640geo_dir.txt')
        self.mat_files = open(self.txtdir, 'r').readlines()
        self.file_num = len(self.mat_files)
        self.rand_state = RandomState(66)
        self.data_dir_metal = '/media/mirlab/hdd2/DeepLesion_metal/test_raw/mask'
        self.data_dir_partial = '/media/mirlab/hdd2/DeepLesion_metal/test_raw/partial_map/correct_img_'
        self.data_dir_mlp = '/media/mirlab/hdd2/DeepLesion_metal/test_raw/mlp_param'
        
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
        Xgt = gt_file['image'][()]
        gt_file.close()
        file = h5py.File(abs_dir, 'r')
        
        #########################################################################
        imag_idx = random.randint(0, 199)
        random_mask_metal = random.randint(0, 9)
        # imag_idx = 0
        # random_mask_metal = 1
        
        M = np.fromfile(os.path.join(self.data_dir_metal, '%03d_%03d.raw'%(imag_idx, random_mask_metal)), dtype=np.float32).reshape([416, 416])
        partial_volume = np.fromfile(os.path.join(self.data_dir_partial + '%03d_%03d.raw'%(imag_idx, random_mask_metal)), dtype=np.float32).reshape([416, 416])
       
        metal_proj = ray_trafo(M)
        metal_proj = np.asarray(metal_proj)
        
        mu = np.load(os.path.join(self.data_dir_mlp,'%03d_%03d.npy'%(imag_idx, random_mask_metal)))

        powers = list(range(5, -1, -1))      # [10, 9, ..., 0]
        
        mu_bh = sum([mu[i] * metal_proj ** p for i, p in enumerate(powers)])
            
        tolerance = 0.04  # hard limit of ±4%
        
        # Step 1: Generate Gaussian noise
        noise = np.random.randn(6) * (tolerance)  # use smaller std to fit under ±3% more often
        
        # Step 2: Clamp to [-0.03, 0.03]
        bounded_noise = np.clip(noise, -tolerance, tolerance)
        
        # Step 3: Apply noise multiplicatively
        mu_perturbed = mu[:6] * (1 + bounded_noise)

        mu_bh_perturbed = sum([mu_perturbed[i] * metal_proj ** p for i, p in enumerate(powers)])
        img_mu_bh_perturbed = FBPOper(mu_bh_perturbed - mu_bh)
        img_mu_bh_perturbed = np.asarray(img_mu_bh_perturbed)
        
        # (img_mu_bh_perturbed).tofile(
        #     os.path.join('/media/mirlab/hdd2/DeepLesion_metal/test_raw/sava_test/img_mu_bh_perturbed.raw')
        # )
        # pdb.set_trace()
        
        Xres = Xgt + img_mu_bh_perturbed + partial_volume
        # (Xres).tofile(
        #     os.path.join('/media/mirlab/hdd2/DeepLesion_metal/test_raw/sava_test/Xres.raw')
        # )
        # pdb.set_trace()
        #########################################################################
        
        
        Xgt = normalize(Xgt, image_get_minmax())
        Xgt = torch.Tensor(Xgt)
        
        Xres = normalize(Xres, image_get_minmax())
        Xres = torch.Tensor(Xres)

        data = {'source': Xgt, 'target': Xres}
        # data = {'source': Xgt, 'target': Xma}   
        #pdb.set_trace()
        
        #if self.transform:
        #    data = self.transform(data)
            
        return data
    
class MARTrainDataset_val_mlp(udata.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.data_dir_input = os.path.join(self.data_dir, 'corr_image')
        self.data_dir_label = os.path.join(self.data_dir, 'Xgt')
        
        lst_input = os.listdir(self.data_dir_input)
        lst_label = os.listdir(self.data_dir_label)
        
        lst_input.sort()
        lst_label.sort()
        
        self.lst_input = lst_input
        self.lst_label = lst_label
        
       
    def __len__(self):
        return len(self.lst_input)

    def __getitem__(self, index):

        Xres = np.fromfile(os.path.join(self.data_dir_input, 'correct_img_' + self.lst_label[index]), dtype=np.float32).reshape([416, 416])
        Xgt = np.fromfile(os.path.join(self.data_dir_label, self.lst_label[index]), dtype=np.float32).reshape([416, 416])
                           
        Xgt = normalize(Xgt, image_get_minmax())
        Xgt = torch.Tensor(Xgt)

        Xres = normalize(Xres, image_get_minmax())
        Xres = torch.Tensor(Xres)

        data = {'source': Xgt, 'target': Xres}   
           
        return data
        
class MARTrainDataset_train(udata.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.dir = data_dir
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
        Xgt = gt_file['image'][()]
        gt_file.close()
        file = h5py.File(abs_dir, 'r')
        Xma= file['ma_CT'][()]
        XLI =file['LI_CT'][()]
        file.close()
        Xgt = normalize(Xgt, image_get_minmax())
        Xgt = torch.Tensor(Xgt)
        XLI = normalize(XLI, image_get_minmax())
        XLI = torch.Tensor(XLI)
        Xma = normalize(Xma, image_get_minmax())
        Xma = torch.Tensor(Xma)
        data = {'source': Xgt, 'target': XLI}
        # data = {'source': Xgt, 'target': Xma}   
        #pdb.set_trace()
        
        #if self.transform:
        #    data = self.transform(data)
            
        return data
    
class MARTrainDataset_test(udata.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.dir = data_dir
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
        Xgt = gt_file['image'][()]
        gt_file.close()
        file = h5py.File(abs_dir, 'r')
        Xma= file['ma_CT'][()]
        XLI =file['LI_CT'][()]
        file.close()
        Xgt = normalize(Xgt, image_get_minmax())
        Xgt = torch.Tensor(Xgt)
        XLI = normalize(XLI, image_get_minmax())
        XLI = torch.Tensor(XLI)
        Xma = normalize(Xma, image_get_minmax())
        Xma = torch.Tensor(Xma)
        data = {'source': Xgt, 'target': XLI}
        # data = {'source': Xgt, 'target': Xma}   
        
        #if self.transform:
        #    data = self.transform(data)
            
        return data
    
class MARTrainDataset_train_sino(udata.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.dir = data_dir
        self.patch_size = 416
        self.txtdir = os.path.join(self.dir, 'train_640geo_dir.txt')
        self.mat_files = open(self.txtdir, 'r').readlines()
        # self.mat_files = open(self.txtdir, 'r').readlines()[:200] # 20% for training in sino_domain
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
        Xgt = gt_file['image'][()]
        gt_file.close()
        file = h5py.File(abs_dir, 'r')
        Sgt = np.asarray(ray_trafo(Xgt))
        Sma = file['ma_sinogram'][()]
        SLI = file['LI_sinogram'][()]
        file.close()
        
        Sgt_pad = np.pad(Sgt,((0,0),(7,8)),mode='edge')
        Sma_pad = np.pad(Sma,((0,0),(7,8)),mode='edge')
        SLI_pad = np.pad(SLI,((0,0),(7,8)),mode='edge')

        Sgt_pad = normalize(Sgt_pad, proj_get_minmax())
        Sgt_pad = torch.Tensor(Sgt_pad)
        
        Sma_pad = normalize(Sma_pad, proj_get_minmax())
        Sma_pad = torch.Tensor(Sma_pad)
        
        SLI_pad = normalize(SLI_pad, proj_get_minmax())
        SLI_pad = torch.Tensor(SLI_pad)
        
        Sma_pad = torch.cat([Sma_pad,SLI_pad],dim=0)
        
        data = {'source': Sgt_pad, 'target': Sma_pad}   

        return data
    
class MARTrainDataset_test_sino(udata.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.dir = data_dir
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
        Xgt = gt_file['image'][()]
        gt_file.close()
        file = h5py.File(abs_dir, 'r')
        Sgt = np.asarray(ray_trafo(Xgt))
        Sma = file['ma_sinogram'][()]
        SLI = file['LI_sinogram'][()]
        file.close()
        
        Sgt_pad = np.pad(Sgt,((0,0),(7,8)),mode='edge')
        Sma_pad = np.pad(Sma,((0,0),(7,8)),mode='edge')
        SLI_pad = np.pad(SLI,((0,0),(7,8)),mode='edge')

        Sgt_pad = normalize(Sgt_pad, proj_get_minmax())
        Sgt_pad = torch.Tensor(Sgt_pad)
        
        Sma_pad = normalize(Sma_pad, proj_get_minmax())
        Sma_pad = torch.Tensor(Sma_pad)
        
        SLI_pad = normalize(SLI_pad, proj_get_minmax())
        SLI_pad = torch.Tensor(SLI_pad)
        
        Sma_pad = torch.cat([Sma_pad,SLI_pad],dim=0)
        
        data = {'source': Sgt_pad, 'target': Sma_pad}   
            
        return data
    
class MARTrainDataset_train2(udata.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.data_dir_input = os.path.join(self.data_dir, 'Xnet')
        self.data_dir_input2 = os.path.join(self.data_dir, 'Xma')
        self.data_dir_label = os.path.join(self.data_dir, 'Xgt')
        
        lst_input = os.listdir(self.data_dir_input)
        lst_input2 = os.listdir(self.data_dir_input2)
        lst_label = os.listdir(self.data_dir_label)
        
        lst_input.sort()
        lst_input2.sort()
        lst_label.sort()
        
        self.lst_input = lst_input
        self.lst_input2 = lst_input2
        self.lst_label = lst_label
        
        self.index_list = np.arange(1000)
        self.rand_state = RandomState(66)
        
    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        imag_idx = self.index_list[index]
        random_mask = random.randint(0, 89)
        random_soft = np.random.rand(1)
        Xnet = np.fromfile(os.path.join(self.data_dir_input, '%03d_%03d.raw'%(imag_idx, random_mask)), dtype=np.float32).reshape([416, 416])
        Xma = np.fromfile(os.path.join(self.data_dir_input2, '%03d_%03d.raw'%(imag_idx, random_mask)), dtype=np.float32).reshape([416, 416])
        Xgt = np.fromfile(os.path.join(self.data_dir_label, '%03d_%03d.raw'%(imag_idx, random_mask)), dtype=np.float32).reshape([416, 416])
        
        Xgt = normalize(Xgt, image_get_minmax())
        Xgt = torch.Tensor(Xgt)

        Xnet = normalize(Xnet, image_get_minmax())
        Xnet = torch.Tensor(Xnet)
        
        Xma = normalize(Xma, image_get_minmax())
        Xma = torch.Tensor(Xma)
        
        # Xnet = torch.cat([Xma, Xnet],dim=0)

        Xsoft = random_soft[0]*Xma + (1-random_soft[0])*Xnet
        data = {'source': Xgt, 'target': Xsoft}   
            
        return data
    
class MARTrainDataset_test2(udata.Dataset):
   def __init__(self, data_dir):
       self.data_dir = data_dir

       self.data_dir_input = os.path.join(self.data_dir, 'Xnet')
       self.data_dir_input2 = os.path.join(self.data_dir, 'Xma')
       self.data_dir_label = os.path.join(self.data_dir, 'Xgt')
       
       lst_input = os.listdir(self.data_dir_input)
       lst_input2 = os.listdir(self.data_dir_input2)
       lst_label = os.listdir(self.data_dir_label)
       
       lst_input.sort()
       lst_input2.sort()
       lst_label.sort()
       
       self.lst_input = lst_input
       self.lst_input2 = lst_input2
       self.lst_label = lst_label
       
       self.index_list = np.arange(200)
       self.rand_state = RandomState(66)
       
   def __len__(self):
       return len(self.index_list)

   def __getitem__(self, index):
       imag_idx = self.index_list[index]
       random_mask = random.randint(0, 9)
       random_soft = np.random.rand(1)
       Xnet = np.fromfile(os.path.join(self.data_dir_input, '%03d_%03d.raw'%(imag_idx, random_mask)), dtype=np.float32).reshape([416, 416])
       Xma = np.fromfile(os.path.join(self.data_dir_input2, '%03d_%03d.raw'%(imag_idx, random_mask)), dtype=np.float32).reshape([416, 416])
       Xgt = np.fromfile(os.path.join(self.data_dir_label, '%03d_%03d.raw'%(imag_idx, random_mask)), dtype=np.float32).reshape([416, 416])
       
       Xgt = normalize(Xgt, image_get_minmax())
       Xgt = torch.Tensor(Xgt)

       Xnet = normalize(Xnet, image_get_minmax())
       Xnet = torch.Tensor(Xnet)
       
       Xma = normalize(Xma, image_get_minmax())
       Xma = torch.Tensor(Xma)
       
       # Xnet = torch.cat([Xma, Xnet],dim=0)
       Xsoft = random_soft[0]*Xma + (1-random_soft[0])*Xnet
       data = {'source': Xgt, 'target': Xsoft}   
           
       return data
   
class MARTrainDataset_train3(udata.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.dir = data_dir
        self.patch_size = 416
        self.txtdir = os.path.join(self.dir, 'train_640geo_dir.txt')
        self.mat_files = open(self.txtdir, 'r').readlines()
        self.file_num = len(self.mat_files)
        self.rand_state = RandomState(66)
    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        gt_dir = self.mat_files[idx]
        random_mask = random.randint(0, 89)  # include 44 part1
        random_soft = np.random.rand(1)
        # random_mask = random.randint(45, 89)  # include 89 part2
        #random_mask = random.randint(0, 9)  # for demo
        file_dir = gt_dir[:-6]
        data_file = file_dir + str(random_mask) + '.h5'
        abs_dir = os.path.join(self.dir, 'train_640geo/', data_file)
        gt_absdir = os.path.join(self.dir,'train_640geo/', gt_dir[:-1])
        gt_file = h5py.File(gt_absdir, 'r')
        Xgt = gt_file['image'][()]
        gt_file.close()
        file = h5py.File(abs_dir, 'r')
        Xma= file['ma_CT'][()]
        XLI =file['LI_CT'][()]
        file.close()
        Xgt = normalize(Xgt, image_get_minmax())
        Xgt = torch.Tensor(Xgt)
        XLI = normalize(XLI, image_get_minmax())
        XLI = torch.Tensor(XLI)
        Xma = normalize(Xma, image_get_minmax())
        Xma = torch.Tensor(Xma)
        # Xma = torch.cat([Xma, XLI],dim=0)
        Xsoft = random_soft[0]*Xma + (1-random_soft[0])*XLI
        data = {'source': Xgt, 'target': Xsoft}   
        #pdb.set_trace()
        
        #if self.transform:
        #    data = self.transform(data)
            
        return data
    
class MARTrainDataset_test3(udata.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.dir = data_dir
        self.patch_size = 416
        self.txtdir = os.path.join(self.dir, 'test_640geo_dir.txt')
        self.mat_files = open(self.txtdir, 'r').readlines()
        self.file_num = len(self.mat_files)
        self.rand_state = RandomState(66)
    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        gt_dir = self.mat_files[idx]
        random_mask = random.randint(0, 9)  # for demo
        random_soft = np.random.rand(1)
        file_dir = gt_dir[:-6]
        data_file = file_dir + str(random_mask) + '.h5'
        abs_dir = os.path.join(self.dir, 'test_640geo/', data_file)
        gt_absdir = os.path.join(self.dir,'test_640geo/', gt_dir[:-1])
        gt_file = h5py.File(gt_absdir, 'r')
        Xgt = gt_file['image'][()]
        gt_file.close()
        file = h5py.File(abs_dir, 'r')
        Xma= file['ma_CT'][()]
        XLI =file['LI_CT'][()]
        file.close()
        Xgt = normalize(Xgt, image_get_minmax())
        Xgt = torch.Tensor(Xgt)
        XLI = normalize(XLI, image_get_minmax())
        XLI = torch.Tensor(XLI)
        Xma = normalize(Xma, image_get_minmax())
        Xma = torch.Tensor(Xma)
        # Xma = torch.cat([Xma, XLI],dim=0)
        Xsoft = random_soft[0]*Xma + (1-random_soft[0])*XLI
        data = {'source': Xgt, 'target': Xsoft}   
        
        #if self.transform:
        #    data = self.transform(data)
            
        return data
    
def test_image_orig(data_path, imag_idx, mask_idx, inner_dir):
    txtdir = os.path.join(data_path, 'train_640geo_dir.txt')
    test_mask = np.load(os.path.join(data_path, 'trainmask.npy'))
    with open(txtdir, 'r') as f:
        mat_files = f.readlines()
    gt_dir = mat_files[imag_idx]
    file_dir = gt_dir[:-6]
    data_file = file_dir + str(mask_idx) + '.h5'
    abs_dir = os.path.join(data_path, inner_dir, data_file)
    gt_absdir = os.path.join(data_path, inner_dir, gt_dir[:-1])
    gt_file = h5py.File(gt_absdir, 'r')
    Xgt = gt_file['image'][()]
    gt_file.close()
    file = h5py.File(abs_dir, 'r')
    Xma = file['ma_CT'][()]
    XLI = file['LI_CT'][()]
    file.close()
    XLI = normalize(XLI, image_get_minmax())
    Xma = normalize(Xma, image_get_minmax())
    Xgt = normalize(Xgt, image_get_minmax())
    
    return torch.Tensor(Xma).cuda(), torch.Tensor(XLI).cuda()
    
def test_image(data_path, imag_idx, mask_idx, inner_dir):
    txtdir = os.path.join(data_path, 'train_640geo_dir.txt')
    #test_mask = np.load(os.path.join(data_path, 'trainmask.npy'))
    rand_state = RandomState(66)
    with open(txtdir, 'r') as f:
        mat_files = f.readlines()
    gt_dir = mat_files[imag_idx]
    file_dir = gt_dir[:-6]
    data_file = file_dir + str(mask_idx) + '.h5'
    abs_dir = os.path.join(data_path, inner_dir, data_file)
    gt_absdir = os.path.join(data_path, inner_dir, gt_dir[:-1])
    gt_file = h5py.File(gt_absdir, 'r')
    Xgt = gt_file['image'][()]
    gt_file.close()
    file = h5py.File(abs_dir, 'r')
    Xma = file['ma_CT'][()]
    XLI = file['LI_CT'][()]
    file.close()
    XLI = normalize(XLI, image_get_minmax())
    Xma = normalize(Xma, image_get_minmax())
    Xgt = normalize(Xgt, image_get_minmax())
    random_soft = 0.3
    # random_soft = np.random.rand(1)
    Xsoft = random_soft*Xma + (1-random_soft)*XLI
    return torch.Tensor(Xsoft).cuda(), torch.Tensor(Xma).cuda()

def test_image2(data_path, imag_idx, mask_idx):
    Xnet = np.fromfile(os.path.join(data_path, 'corr_image/correct_img_%03d_%03d.raw'%(imag_idx, mask_idx)), dtype=np.float32).reshape([416, 416])
    # Xma = np.fromfile(os.path.join(data_path, 'Xma/%03d_%03d.raw'%(imag_idx, mask_idx)), dtype=np.float32).reshape([416, 416])
    Xgt = np.fromfile(os.path.join(data_path, 'Xgt/%03d_%03d.raw'%(imag_idx, mask_idx)), dtype=np.float32).reshape([416, 416])
    
    Xgt = normalize(Xgt, image_get_minmax())
    Xgt = torch.Tensor(Xgt)

    Xnet = normalize(Xnet, image_get_minmax())
    Xnet = torch.Tensor(Xnet)
    
    # Xma = normalize(Xma, image_get_minmax())
    # Xma = torch.Tensor(Xma)
    
    # Xnet = torch.cat([Xma, Xnet],dim=0)
    # Xsoft = 0.5*Xma + 0.5*Xnet

    return torch.Tensor(Xnet).cuda(), torch.Tensor(Xgt).cuda()

def test_image_clinic(data_path, imag_idx, mask_idx):
    Xnet = np.fromfile(os.path.join(data_path, 'corr_image_clinic/correct_img_0192.raw'), dtype=np.float32).reshape([416, 416])
    # Xma = np.fromfile(os.path.join(data_path, 'Xma/%03d_%03d.raw'%(imag_idx, mask_idx)), dtype=np.float32).reshape([416, 416])
    # Xgt = np.fromfile(os.path.join(data_path, 'Xgt/%03d_%03d.raw'%(imag_idx, mask_idx)), dtype=np.float32).reshape([416, 416])
    
    # Xgt = normalize(Xgt, image_get_minmax())
    # Xgt = torch.Tensor(Xgt)

    Xnet = normalize(Xnet, image_get_minmax())
    Xnet = torch.Tensor(Xnet)
    
    # Xma = normalize(Xma, image_get_minmax())
    # Xma = torch.Tensor(Xma)
    
    # Xnet = torch.cat([Xma, Xnet],dim=0)
    # Xsoft = 0.5*Xma + 0.5*Xnet

    return torch.Tensor(Xnet).cuda()

def test_image_sino(data_path, imag_idx, mask_idx, inner_dir):
    txtdir = os.path.join(data_path, 'test_640geo_dir.txt')
    test_mask = np.load(os.path.join(data_path, 'testmask.npy'))
    with open(txtdir, 'r') as f:
        mat_files = f.readlines()
        # mat_files = f.readlines()[800:]
    gt_dir = mat_files[imag_idx]
    file_dir = gt_dir[:-6]
    data_file = file_dir + str(mask_idx) + '.h5'
    abs_dir = os.path.join(data_path, inner_dir, data_file)
    gt_absdir = os.path.join(data_path, inner_dir, gt_dir[:-1])
    gt_file = h5py.File(gt_absdir, 'r')
    Xgt = gt_file['image'][()]
    gt_file.close()
    file = h5py.File(abs_dir, 'r')
    Sgt = np.asarray(ray_trafo(Xgt))
    Sma = file['ma_sinogram'][()]
    SLI = file['LI_sinogram'][()]
    Xma= file['ma_CT'][()]
    file.close()
    
    SLI_pad = np.pad(SLI,((0,0),(7,8)),mode='edge')
    Sgt_pad = np.pad(Sgt,((0,0),(7,8)),mode='edge')
    Sma_pad = np.pad(Sma,((0,0),(7,8)),mode='edge')

    Sgt_pad = normalize(Sgt_pad, proj_get_minmax())
    Sgt_pad = torch.Tensor(Sgt_pad)
    
    Sma_pad = normalize(Sma_pad, proj_get_minmax())
    Sma_pad = torch.Tensor(Sma_pad)
    
    SLI_pad = normalize(SLI_pad, proj_get_minmax())
    SLI_pad = torch.Tensor(SLI_pad)
    
    Sma_pad = torch.cat([Sma_pad,SLI_pad],dim=0)
    
    return torch.Tensor(Sma_pad).cuda(), torch.Tensor(Sgt_pad).cuda(), torch.Tensor(Xgt).cuda()

import scipy.io as sio
from sklearn.cluster import k_means
import scipy
sigma = 1
smFilter = sio.loadmat('deeplesion/gaussianfilter.mat')['smFilter']
miuAir = 0
miuWater=0.192
starpoint = np.zeros([3, 1])
starpoint[0] = miuAir
starpoint[1] = miuWater
starpoint[2] = 2 * miuWater
def nmarprior(im,threshWater,threshBone,miuAir,miuWater,smFilter):
    imSm = scipy.ndimage.filters.convolve(im, smFilter, mode='nearest')
    # print("imSm, h:, w:", imSm.shape[0], imSm.shape[1]) # imSm, h:, w: 416 416
    priorimgHU = imSm
    priorimgHU[imSm <= threshWater] = miuAir
    h, w = imSm.shape[0], imSm.shape[1]
    priorimgHUvector = np.reshape(priorimgHU, h*w)
    region1_1d = np.where(priorimgHUvector > threshWater)
    region2_1d = np.where(priorimgHUvector < threshBone)
    region_1d = np.intersect1d(region1_1d, region2_1d)
    priorimgHUvector[region_1d] = miuWater
    priorimgHU = np.reshape(priorimgHUvector,(h,w))
    return priorimgHU

def nmar_prior(XLI):
    # XLI[M == 1] = 0.192
    h, w = XLI.shape[0], XLI.shape[1]
    im1d = XLI.reshape(h * w, 1)
    best_centers, labels, best_inertia = k_means(im1d, n_clusters=3, init=starpoint, max_iter=300)
    threshBone2 = np.min(im1d[labels ==2])
    threshBone2 = np.max([threshBone2, 1.2 * miuWater])
    threshWater2 = np.min(im1d[labels == 1])
    imPriorNMAR = nmarprior(XLI, threshWater2, threshBone2, miuAir, miuWater, smFilter)
    return imPriorNMAR

def test_image_NMAR(data_path, imag_idx, mask_idx, inner_dir):
    txtdir = os.path.join(data_path, 'test_640geo_dir.txt')
    test_mask = np.load(os.path.join(data_path, 'testmask.npy'))
    with open(txtdir, 'r') as f:
        mat_files = f.readlines()
        # mat_files = f.readlines()[200:]
    gt_dir = mat_files[imag_idx]
    file_dir = gt_dir[:-6]
    data_file = file_dir + str(mask_idx) + '.h5'
    abs_dir = os.path.join(data_path, inner_dir, data_file)
    gt_absdir = os.path.join(data_path, inner_dir, gt_dir[:-1])
    gt_file = h5py.File(gt_absdir, 'r')
    Xgt = gt_file['image'][()]
    gt_file.close()
    file = h5py.File(abs_dir, 'r')
    Sma = file['ma_sinogram'][()]
    Tr = file['metal_trace'][()]
    file.close()
    
    Xnet = np.fromfile(os.path.join('/data/1/medusion-deeplesion/scripts/results/deeplesion_mlp/Xout/%03d_%03d.raw'%(imag_idx, mask_idx)), dtype=np.float32).reshape([416, 416])
    # Xnet_prior = nmar_prior(Xnet)
    Sprior = np.asarray(ray_trafo(Xnet))
    
    Snorm = (Sma+1)/(Sprior+1)
    
    Snorm = interpolate_projection(Snorm, Tr)
    Sdenorm = Snorm*(Sprior+1)-1
    # Sdenorm = Sma*(1-Tr) + Sprior*Tr
    Sdenorm = torch.Tensor(Sdenorm)
    
    M = test_mask[:,:,mask_idx]
    M = np.array(Image.fromarray(M).resize((416, 416), PIL.Image.BILINEAR))
    
    return torch.Tensor(Sdenorm).cuda(), M

def test_image_NMAR_clinic(data_path, imag_idx, mask_idx, inner_dir):
    txtdir = os.path.join(data_path, 'test_640geo_dir.txt')
    test_mask = np.load(os.path.join(data_path, 'testmask.npy'))
    with open(txtdir, 'r') as f:
        mat_files = f.readlines()
        # mat_files = f.readlines()[200:]
    gt_dir = mat_files[imag_idx]
    file_dir = gt_dir[:-6]
    data_file = file_dir + str(mask_idx) + '.h5'
    abs_dir = os.path.join(data_path, inner_dir, data_file)
    gt_absdir = os.path.join(data_path, inner_dir, gt_dir[:-1])
    gt_file = h5py.File(gt_absdir, 'r')
    Xgt = gt_file['image'][()]
    gt_file.close()
    file = h5py.File(abs_dir, 'r')
    # Sma = file['ma_sinogram'][()]
    # Tr = file['metal_trace'][()]
    file.close()
    
    Tr = np.fromfile(os.path.join('/media/mirlab/hdd2/DeepLesion_metal/test_raw/corr_image_clinic/0192.raw'), dtype=np.float32).reshape([640, 641])
    Sma = np.fromfile(os.path.join('/media/mirlab/hdd2/DeepLesion_metal/test_raw/corr_image_clinic/Sma_0192.raw'), dtype=np.float32).reshape([640, 641])
    Xnet = np.fromfile(os.path.join('/data/1/medusion-deeplesion/scripts/results/deeplesion_mlp/Xout_clinic/correct_img_0192.raw'), dtype=np.float32).reshape([416, 416])
    # Xnet_prior = nmar_prior(Xnet)
    Sprior = np.asarray(ray_trafo(Xnet))
    
    Snorm = (Sma+1)/(Sprior+1)
    
    Snorm = interpolate_projection(Snorm, Tr)
    Sdenorm = Snorm*(Sprior+1)-1
    # Sdenorm = Sma*(1-Tr) + Sprior*Tr
    Sdenorm = torch.Tensor(Sdenorm)
    
    M = test_mask[:,:,mask_idx]
    M = np.array(Image.fromarray(M).resize((416, 416), PIL.Image.BILINEAR))
    
    return torch.Tensor(Sdenorm).cuda(), M

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

class MARTrainDataset_train_progressive(udata.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.data_dir_input = os.path.join(self.data_dir, 'Xout')
        # self.data_dir_input2 = os.path.join(self.data_dir, 'Xma')
        self.data_dir_label = os.path.join(self.data_dir, 'Xgt')
        
        lst_input = os.listdir(self.data_dir_input)
        # lst_input2 = os.listdir(self.data_dir_input2)
        lst_label = os.listdir(self.data_dir_label)
        
        lst_input.sort()
        # lst_input2.sort()
        lst_label.sort()
        
        self.lst_input = lst_input
        # self.lst_input2 = lst_input2
        self.lst_label = lst_label
        
        self.index_list = np.arange(1000)
        self.rand_state = RandomState(66)
        
    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        imag_idx = self.index_list[index]
        # random_mask = random.randint(0, 89)
        random_mask = random.randint(45, 89)
        Xnet = np.fromfile(os.path.join(self.data_dir_input, '%03d_%03d.raw'%(imag_idx, random_mask)), dtype=np.float32).reshape([416, 416])
        # Xma = np.fromfile(os.path.join(self.data_dir_input2, '%03d_%03d.raw'%(imag_idx, random_mask)), dtype=np.float32).reshape([416, 416])
        Xgt = np.fromfile(os.path.join(self.data_dir_label, '%03d_%03d.raw'%(imag_idx, random_mask)), dtype=np.float32).reshape([416, 416])
        
        Xgt = normalize(Xgt, image_get_minmax())
        Xgt = torch.Tensor(Xgt)

        Xnet = normalize(Xnet, image_get_minmax())
        Xnet = torch.Tensor(Xnet)
        
        # Xma = normalize(Xma, image_get_minmax())
        # Xma = torch.Tensor(Xma)
        
        # Xnet = torch.cat([Xma, Xnet],dim=0)

        data = {'source': Xgt, 'target': Xnet}   
            
        return data
    
class MARTrainDataset_test_progressive(udata.Dataset):
   def __init__(self, data_dir):
       self.data_dir = data_dir

       self.data_dir_input = os.path.join(self.data_dir, 'Xout')
       # self.data_dir_input2 = os.path.join(self.data_dir, 'Xma')
       self.data_dir_label = os.path.join(self.data_dir, 'Xgt')
       
       lst_input = os.listdir(self.data_dir_input)
       # lst_input2 = os.listdir(self.data_dir_input2)
       lst_label = os.listdir(self.data_dir_label)
       
       lst_input.sort()
       # lst_input2.sort()
       lst_label.sort()
       
       self.lst_input = lst_input
       # self.lst_input2 = lst_input2
       self.lst_label = lst_label
       
       self.index_list = np.arange(200)
       self.rand_state = RandomState(66)
       
   def __len__(self):
       return len(self.index_list)

   def __getitem__(self, index):
       imag_idx = self.index_list[index]
       random_mask = random.randint(0, 9)
       
       Xnet = np.fromfile(os.path.join(self.data_dir_input, '%03d_%03d.raw'%(imag_idx, random_mask)), dtype=np.float32).reshape([416, 416])
       # Xma = np.fromfile(os.path.join(self.data_dir_input2, '%03d_%03d.raw'%(imag_idx, random_mask)), dtype=np.float32).reshape([416, 416])
       Xgt = np.fromfile(os.path.join(self.data_dir_label, '%03d_%03d.raw'%(imag_idx, random_mask)), dtype=np.float32).reshape([416, 416])
       
       Xgt = normalize(Xgt, image_get_minmax())
       Xgt = torch.Tensor(Xgt)

       Xnet = normalize(Xnet, image_get_minmax())
       Xnet = torch.Tensor(Xnet)
       
       # Xma = normalize(Xma, image_get_minmax())
       # Xma = torch.Tensor(Xma)
       
       # Xnet = torch.cat([Xma, Xnet],dim=0)

       data = {'source': Xgt, 'target': Xnet}   
           
       return data