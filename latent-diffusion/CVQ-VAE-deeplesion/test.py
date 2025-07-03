import os
import argparse
import multiprocessing as mp
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
from torchvision import transforms, datasets

from modules import Model
from util import tensor2im, save_image
from dataset_test import *
from torch.utils.data import DataLoader
from skimage import io
import struct
import os
parser = argparse.ArgumentParser(description='CVQ-VAE')
# General
parser.add_argument('--data_folder', type=str, help='name of the data folder')
parser.add_argument('--dataset', type=str, help='name of the dataset (mnist, fashion-mnist, cifar10)')
parser.add_argument('--batch_size', type=int, default=16, help='batch size (default: 16)')
# Latent space
parser.add_argument('--hidden_size', type=int, default=128, help='size of the latent vectors (default: 128)')
parser.add_argument('--num_residual_hidden', type=int, default=32, help='size of the redisual layers (default: 32)')
parser.add_argument('--num_residual_layers', type=int, default=2, help='number of residual layers (default: 2)')
# Quantiser parameters
parser.add_argument('--embedding_dim', type=int, default=64, help='dimention of codebook (default: 64)')
parser.add_argument('--num_embedding', type=int, default=512, help='number of codebook (default: 512)')
parser.add_argument('--distance', type=str, default='cos', help='distance for codevectors and features')
parser.add_argument('--perceptual_loss', action='store_true', help='using perceptual loss')
# Miscellaneous
parser.add_argument('--output_folder', type=str, default='/scratch/shared/beegfs/cxzheng/normcode/final_vqvae/', help='name of the output folder (default: vqvae)')
parser.add_argument('--model_name', type=str, default='fashionmnist_probrandom_contramin1/best.pt', help='name of the output folder (default: vqvae)')
parser.add_argument('--num_workers', type=int, default=mp.cpu_count() - 1, help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
parser.add_argument('--device', type=str, default='cuda', help='set the device (cpu or cuda, default: cpu)')
args = parser.parse_args()


# load dataset
#transform = transforms.Compose([
#            transforms.ToTensor(),
#            transforms.Normalize((0.5), (0.5))
#    ])
if args.dataset == 'mnist':
    # Define the train & test datasets
    test_dataset = datasets.MNIST(args.data_folder, train=False, download=True, transform=transform)
    num_channels = 1
elif args.dataset == 'fashion-mnist':
    # Define the train & test datasets
    test_dataset = datasets.FashionMNIST(args.data_folder, train=False, download=True, transform=transform)
    num_channels = 1
elif args.dataset == 'cifar10':
    # Define the train & test datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CIFAR10(args.data_folder, train=False, download=True, transform=transform)
    num_channels = 3
elif args.dataset == 'celeba':
    # Define the train & test datasets
    transform = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CelebA(args.data_folder, split='valid', download=True, transform=transform)
    num_channels = 3

# Define the dataloaders
transform = transforms.Compose([ToTensor()])
dataset_test = Dataset(os.path.join(args.data_folder, 'val'), nu=512, nv=512, transform=transform)
test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

num_channels = 1 # Input Channel

# Define loaded dir file name
lst_input = os.listdir(os.path.join(args.data_folder, 'val/input'))
lst_input.sort()
lst_input = lst_input  

# Define the model
model = Model(num_channels, args.hidden_size, args.num_residual_layers, args.num_residual_hidden,
                  args.num_embedding, args.embedding_dim, distance=args.distance, perceiver=args.perceptual_loss)

# load model
ckpt = torch.load(os.path.join(os.path.join(os.path.join(args.output_folder, 'models'), args.model_name)))
model.load_state_dict(ckpt)
model = model.to(args.device)
model.eval()

# store results
results_path = os.path.join(os.path.join(args.output_folder, 'results'), args.model_name)
original_path = os.path.join(results_path, 'original')
rec_path = os.path.join(results_path, 'rec')
if not os.path.exists(results_path):
    os.makedirs(original_path)
    os.makedirs(rec_path)

# test model
encodings = []
indexes = []
all_images = []
imageid = 0

# Tensor to Cpu
fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
id=0
for images in test_loader:
    images = images['input'].to(args.device)
    x_recons, _, _, _, _ = model(images)
    #x_recons, loss, perplexity, encoding = model(images)
    # save indexes
    #index = encoding.argmax(dim=1).view(images.size(0), -1)
    #indexes.append(index)
    #all_images.append(images.view(images.size(0), -1))
    # save encodings
    #encodings.append(encoding)
    # save image
    for x_recon, image in zip(x_recons, images):
        x_recon = fn_tonumpy(x_recon)
        #x_recon = ((x_recon+1)/2)*6500-1500
        x_recon = ((x_recon+1)/2)*0.15
        
        f = open(os.path.join(rec_path, lst_input[id]), "wb")
        output_p = np.reshape(x_recon[0].squeeze(), 512*512)
        myfmt = 'f' * len(output_p)
        bin = struct.pack(myfmt, *output_p)
        f.write(bin)
        f.close
        
        image = fn_tonumpy(image)
        #image = ((image+1)/2)*6500-1500
        image = ((image+1)/2)*0.15
        
        f = open(os.path.join(original_path, lst_input[id]), "wb")
        output_p = np.reshape(image[0].squeeze(), 512*512)
        myfmt = 'f' * len(output_p)
        bin = struct.pack(myfmt, *output_p)
        f.write(bin)
        f.close
        
        id = id + 1
