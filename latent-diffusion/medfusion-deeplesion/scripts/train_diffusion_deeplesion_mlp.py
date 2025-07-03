
from email.mime import audio
from pathlib import Path
from datetime import datetime

import torch 
import torch.nn as nn
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np 
import torchio as tio 

from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import MARTrainDataset_train_mlp, MARTrainDataset_val_mlp, AIROGSDataset, MSIvsMSS_2_Dataset, CheXpert_2_Dataset
from medical_diffusion.models.pipelines import DiffusionPipeline
from medical_diffusion.models.estimators import UNet
from medical_diffusion.external.stable_diffusion.unet_openai import UNetModel
from medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from medical_diffusion.models.embedders import LabelEmbedder_deeplesion, TimeEmbbeding
from medical_diffusion.models.embedders.latent_embedders import VAE, VAEGAN, CVQVAE, VQGAN

import torch.multiprocessing as mp
from torchvision import transforms, datasets

from cvqvae.modules import Model

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    mp.set_sharing_strategy('file_system')
    # ------------ Load Data ----------------
    ds = MARTrainDataset_train_mlp( #  256x256
        data_dir='/media/mirlab/hdd2/DeepLesion_metal/train', 
    )
    
    ds_val = MARTrainDataset_val_mlp(  # 256x256
        data_dir='/media/mirlab/hdd2/DeepLesion_metal/test_raw',
    )
    
    dm = SimpleDataModule(
        ds_train = ds,
        ds_val=ds_val,
        batch_size = 8, 
        num_workers=0,
        pin_memory = True,
        # weights=ds.get_weights()   
    ) 
    
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    # path_run_dir = Path.cwd() / 'runs/diffusion_deeplesion_Xma/' / str(current_time)
    path_run_dir = Path('/home/mirlab/Desktop/AAPM_Challanges/deeplesion_checkpoint/diffusion_deeplesion_mlp/') / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'



    # ------------ Initialize Model ------------
    #cond_embedder = None 
    cond_embedder = LabelEmbedder_deeplesion
    cond_embedder_kwargs = {
        'emb_dim': 1024,
        'num_classes': 2
    }
 
    time_embedder = TimeEmbbeding
    time_embedder_kwargs ={
        'emb_dim': 1024 # stable diffusion uses 4*model_channels (model_channels is about 256)
    }


    noise_estimator = UNet
    noise_estimator_kwargs = {
        'in_ch':4, 
        'out_ch':4, 
        'spatial_dims':2,
        'hid_chs':  [  256, 256, 512, 1024],
        'kernel_sizes':[3, 3, 3, 3],
        'strides':     [1, 2, 2, 2],
        'time_embedder':time_embedder,
        'time_embedder_kwargs': time_embedder_kwargs,
        'cond_embedder':cond_embedder,
        'cond_embedder_kwargs': cond_embedder_kwargs,
        'deep_supervision': False,
        'use_res_block':True,
        'use_attention':'none',
    }


    # ------------ Initialize Noise ------------
    noise_scheduler = GaussianNoiseScheduler
    noise_scheduler_kwargs = {
        'timesteps': 1000,
        'beta_start': 0.002, # 0.0001, 0.0015
        'beta_end': 0.02, # 0.01, 0.0195
        'schedule_strategy': 'scaled_linear'
    }
    
    # ------------ Initialize Latent Space  ------------
    latent_embedder = Model(input_dim=1, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, num_embeddings=4096, embedding_dim=4, distance='cos', perceiver=True)
    latent_embedder_checkpoint = '/media/mirlab/hdd2/CVQ-VAE-deeplesion/results/Deeplesion_128_4096_4/models/Xma/model_1000.pt'
   
    # ------------ Initialize Pipeline ------------
    pipeline = DiffusionPipeline(
        noise_estimator=noise_estimator, 
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=noise_scheduler, 
        noise_scheduler_kwargs = noise_scheduler_kwargs,
        latent_embedder=latent_embedder,
        latent_embedder_checkpoint = latent_embedder_checkpoint,
        estimator_objective='x_0',
        estimate_variance=False, 
        use_self_conditioning=False, 
        use_ema=True,
        classifier_free_guidance_dropout=0, # Disable during training by setting to 0
        do_input_centering=False,
        clip_x0=False,
        sample_every_n_steps=376*5,
        loss=torch.nn.MSELoss,
        scale_factor=1.0,
        num_samples = 1,
        optimizer_kwargs={'lr':1e-4}
    )
    
    #------------Load Checkpoint-------------------
    #pipeline_old = pipeline.load_from_checkpoint('runs/diffusion_v3/2024_05_21_235828/last.ckpt')
    #pipeline.noise_estimator.load_state_dict(pipeline_old.noise_estimator.state_dict(), strict=True)

    # -------------- Training Initialization ---------------
    to_monitor = "val/loss"  # "pl/val_loss" 
    min_max = "min"
    save_and_sample_every = 100

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
        patience=30, # number of checks with no improvement
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir), # dirpath
        monitor=to_monitor,
        every_n_train_steps=save_and_sample_every,
        save_last=True,
        save_top_k=5,
        mode=min_max,
    )
    trainer = Trainer(
        accelerator=accelerator,
        # devices=[0],
        # precision=16,
        # amp_backend='apex',
        # amp_level='O2',
        # gradient_clip_val=0.5,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        # callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=save_and_sample_every, 
        auto_lr_find=False,
        # limit_train_batches=1000,
        limit_val_batches=1.0, # 0 = disable validation - Note: Early Stopping no longer available 
        min_epochs=1,
        max_epochs=500,
        num_sanity_val_steps=2,
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(pipeline, datamodule=dm)

    # ------------- Save path to best model -------------
    pipeline.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)


