import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from pathlib import Path
import random
from datetime import datetime

from matplotlib import pyplot as plt

from model import VPTREnc, VPTRDec, VPTRDisc, init_weights
from model import GDL, MSELoss, L1Loss, GANLoss
from utils import KTHDataset, BAIRDataset, MovingMNISTDataset
from utils import VidCenterCrop, VidPad, VidResize, VidNormalize, VidReNormalize, VidCrop, VidRandomHorizontalFlip, VidRandomVerticalFlip, VidToTensor
from utils import visualize_batch_clips, save_ckpt, load_ckpt, set_seed, AverageMeters, init_loss_dict, write_summary, resume_training
from utils import set_seed, get_dataloader

set_seed(2021)


resume_ckpt = Path('scratch/ms14625/VTPR/VPTR_ckpts/ae_run_blocks_MSEGDLgan_ckpt/epoch_1.tar') #specify your trained autoencoder checkpoint file
num_past_frames = 10
num_future_frames = 10
encH, encW, encC = 8, 8, 64
img_channels = 1 #Set to be 3 for BAIR dataset
N = 16
ngf = 32 

device = torch.device('cuda')
loss_name_list = ['AE_L1', 'Dtotal', 'Dfake', 'Dreal', 'AEgan']

VPTR_Enc = VPTREnc(img_channels, ngf=ngf, feat_dim = encC, n_downsampling = 3).to(device) #Set the padding_type to be "zero" for BAIR dataset

#Set the padding_type to be "zero" for BAIR dataset, set the out_layer to be 'Sigmoid' for MovingMNIST
VPTR_Dec = VPTRDec(img_channels, ngf=ngf, feat_dim = encC, n_downsampling = 3, out_layer = 'Tanh').to(device) 

init_weights(VPTR_Enc)
init_weights(VPTR_Dec)

loss_dict, start_epoch = resume_training({'VPTR_Enc': VPTR_Enc, 'VPTR_Dec': VPTR_Dec}, 
                                         {}, resume_ckpt, loss_name_list)


data_set_name = 'BLOCKS' #see utils.dataset
# dataset_dir = '/home/mrunal/Documents/NYUCourses/DeepLearning/project/VPTR/data/blocks/dataset/unlabeled'
dataset_dir = '/scratch/ms14625/VTPR/data/blocks/dataset/unlabeled'

_, _, test_loader, renorm_transform = get_dataloader(data_set_name, N, dataset_dir, test_past_frames = 10, test_future_frames = 10, ngpus = 1, num_workers = 1)

model = nn.Sequential(VPTR_Enc, VPTR_Dec)

sample = next(iter(test_loader))
past_frames, future_frames = sample
past_frames = past_frames.to(device)
future_frames = future_frames.to(device)
x = torch.cat([past_frames, future_frames], dim = 1 )
pred = model(x)

def plot_model_result(pred, fig_name, num_frames, n = 2):
    fig, ax = plt.subplots(1, num_frames, figsize = (num_frames, 1))
    fig.subplots_adjust(wspace=0., hspace = 0.)

    for j in range(num_frames):
        ax[j].set_axis_off()
        
        img = pred[:, j, :, :, :].clone()
        img = renorm_transform(img)
        img = torch.clamp(img, min = 0., max = 1.)
        img = img[n, ...]

        img = transforms.ToPILImage()(img)
        ax[j].imshow(img, cmap = 'gray')
    fig.savefig(f'{fig_name}.pdf', bbox_inches = 'tight')

plot_model_result(x, 'AE_gt', 20, n = 1)

plot_model_result(pred, 'AE_rec', 20, n = 1)