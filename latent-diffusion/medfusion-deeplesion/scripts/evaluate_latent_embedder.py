from pathlib import Path 
import logging
from datetime import datetime
from tqdm import tqdm

import numpy as np 
import torch
import torchvision.transforms.functional as tF
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, Subset

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.functional import multiscale_structural_similarity_index_measure as mmssim

from medical_diffusion.models.embedders.latent_embedders import CVQVAE, VQGAN

from torchvision import transforms
from medical_diffusion.data.datasets import SH_Dataset_latent_emb, ToTensor
from skimage import io
import os
import pdb
# ----------------Settings --------------
batch_size = 2
max_samples = None # set to None for all 
target_class = None # None for no specific class 
# path_out = Path.cwd()/'results'/'MSIvsMSS_2'/'metrics'
# path_out = Path.cwd()/'results'/'AIROGS'/'metrics'
path_out = Path.cwd()/'results'/'AAPM_CTimage'/'metrics'
path_out.mkdir(parents=True, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------- Logging -----------
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.addHandler(logging.FileHandler(path_out/f'metrics_{current_time}.log', 'w'))


# -------------- Helpers ---------------------
#pil2torch = lambda x: torch.as_tensor(np.array(x)).moveaxis(-1, 0) # In contrast to ToTensor(), this will not cast 0-255 to 0-1 and destroy uint8 (required later)
fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
# ---------------- Dataset/Dataloader ----------------
#ds_real = ImageFolder('/home/ysh/Desktop/AAPM_challange/img_domain/datasets_img/train/', transform=pil2torch)
# ds_real = ImageFolder('/mnt/hdd/datasets/eye/AIROGS/data_256x256_ref/', transform=pil2torch)
# ds_real = ImageFolder('/mnt/hdd/datasets/chest/CheXpert/ChecXpert-v10/reference_test/', transform=pil2torch)

ds_real = SH_Dataset_latent_emb( #  256x256
    data_dir='../data/AAPM_MAR_challange/latent_emb/test', 
    nu=512,
    nv=512,
    transform = transforms.Compose([ToTensor()])
)
# ---------- Limit Sample Size 
#ds_real.samples = ds_real.samples[slice(max_samples)]


# --------- Select specific class ------------
if target_class is not None:
    ds_real = Subset(ds_real, [i for i in range(len(ds_real)) if ds_real.samples[i][1] == ds_real.class_to_idx[target_class]])
dm_real = DataLoader(ds_real, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)

logger.info(f"Samples Real: {len(ds_real)}")

# --------------- Load Model ------------------
model = CVQVAE.load_from_checkpoint('runs/cvqvae/2024_02_21_061643/last.ckpt')
model.to(device)

# from diffusers import StableDiffusionPipeline
# with open('auth_token.txt', 'r') as file:
#     auth_token = file.read()
# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32,  use_auth_token=auth_token)
# model = pipe.vae
# model.to(device)


# ------------- Init Metrics ----------------------
#calc_lpips = LPIPS().to(device)


# --------------- Start Calculation -----------------
mmssim_list, mse_list = [], []
id = 0 
for real_batch in tqdm(dm_real):
    imgs_real_batch = real_batch['source'].to(device)

    #imgs_real_batch = tF.normalize(imgs_real_batch/255, 0.5, 0.5) # [0, 255] -> [-1, 1]
    with torch.no_grad():
        #imgs_fake_batch = model(imgs_real_batch)[0].clamp(-1, 1)
        imgs_fake_batch = model(imgs_real_batch)[0]

    # -------------- LPIP -------------------
    #calc_lpips.update(imgs_real_batch, imgs_fake_batch) # expect input to be [-1, 1]

    # -------------- MS-SSIM + MSE -------------------
    for img_real, img_fake in zip(imgs_real_batch, imgs_fake_batch):
        #img_real, img_fake = (img_real+1)/2, (img_fake+1)/2  # [-1, 1] -> [0, 1]
        #mmssim_list.append(mmssim(img_real[None], img_fake[None], normalize='relu')) 
        #mse_list.append(torch.mean(torch.square(img_real-img_fake)))
        #pdb.set_trace()
    # --------------- Save Image -----------------
        img_fake_cpu = fn_tonumpy(img_fake)
        io.imsave(os.path.join('sample_embedder/output/', 'sample_%04d.tiff' % id),img_fake_cpu[0])
        img_real_cpu = fn_tonumpy(img_real)
        io.imsave(os.path.join('sample_embedder/label/', 'sample_%04d.tiff' % id),img_real_cpu[0])
        id = id + 1
# -------------- Summary -------------------
#mmssim_list = torch.stack(mmssim_list)
#mse_list = torch.stack(mse_list)

#lpips = 1-calc_lpips.compute()
#logger.info(f"LPIPS Score: {lpips}")
#logger.info(f"MS-SSIM: {torch.mean(mmssim_list)} ± {torch.std(mmssim_list)}")
#logger.info(f"MSE: {torch.mean(mse_list)} ± {torch.std(mse_list)}")