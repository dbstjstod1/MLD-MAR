# MLD-MAR
This repository contains two uploaded files:

1. MLP-based beam-hardening correction

2. Latent diffusion model

- The folder includes the latent embedder cvq-vae and the diffusion model medfusion-deeplesion.

The dataset used is available here:
https://github.com/hongwang01/SynDeepLesion

The required virtual environment is based on MedFusion (used to develop the latent diffusion model):
https://github.com/mueller-franzes/medfusion

Please refer to these repositories for data and environment setup.

The model can be trained by using the code at MLD-LDM/latent-diffusion/medfusion-deeplesion/scripts/train_diffusion_deeplesion_mlp.py

The model can be tested by using the code at MLD-LDM/latent-diffusion/medfusion-deeplesion/scripts/sample_deeplesion_mlp.py

The model checkpoint for running the latent diffusion model can be downloaded at https://drive.google.com/file/d/1EGuv2YWvZq3SqYIr9vjiWyw6a4ADK2UG/view?usp=sharing

The checkpoint file has to be placed at MLD-LDM/latent-diffusion/medfusion-deeplesion-checkpoint/diffusion_deeplesion_mlp/2025_04_02_121300


The results also can be downloaded by the link https://drive.google.com/file/d/14Z5Z8aOIZH4eWjUTeAi8igZWyjhylxRX/view?usp=sharing
