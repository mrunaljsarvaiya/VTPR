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
import time

from model import VPTREnc, VPTRDec, VPTRDisc, init_weights, VPTRFormerFAR, VPTRFormerNAR
from model import GDL, MSELoss, L1Loss, GANLoss
from utils import KTHDataset, BAIRDataset, MovingMNISTDataset
from utils import VidCenterCrop, VidPad, VidResize, VidNormalize, VidReNormalize, VidCrop, VidRandomHorizontalFlip, VidRandomVerticalFlip, VidToTensor
from utils import visualize_batch_clips, save_ckpt, load_ckpt, set_seed, AverageMeters, init_loss_dict, write_summary, resume_training, write_code_files
from utils import set_seed, PSNR, SSIM, MSEScore, get_dataloader
import numpy as np


from matplotlib import pyplot as plt

set_seed(2021)

resume_ckpt = Path('/scratch/ms14625/VTPR/VPTR_ckpts/blocks_FAR_kaiming_ckpt/epoch_3.tar') #The trained Transformer checkpoint file
resume_AE_ckpt = Path('/scratch/ms14625/VTPR/VPTR_ckpts/blocks_past_10_future_11_kaiming_ckpt/epoch_28.tar') #The trained AutoEncoder checkpoint file
num_past_frames = 10
num_future_frames = 11
encH, encW, encC = 8, 8, 128
TSLMA_flag = False
rpe = True

img_channels = 3 # 1 for KTH and MovingMNIST, 3 for BAIR
N = 1
ngf = 64
device = torch.device('cpu')
loss_name_list = ['T_MSE', 'T_GDL', 'T_gan', 'T_total', 'Dtotal', 'Dfake', 'Dreal']

#Set the padding_type to be "zero" for BAIR dataset
VPTR_Enc = VPTREnc(img_channels, ngf=ngf, feat_dim = encC, n_downsampling = 3).to(device)

#Set the padding_type to be "zero" for BAIR dataset, set the out_layer to be 'Sigmoid' for MovingMNIST
VPTR_Dec = VPTRDec(img_channels, ngf=ngf, feat_dim = encC, n_downsampling = 3, out_layer = 'Tanh').to(device) #Sigmoid for MNIST, Tanh for KTH and BAIR
VPTR_Enc = VPTR_Enc.eval()
VPTR_Dec = VPTR_Dec.eval()
# VPTR_Transformer = VPTRFormerFAR(num_past_frames, num_future_frames, encH=20, encW = 30, d_model=encC, 
#                                     nhead=8, num_encoder_layers=12, dropout=0.1, 
#                                     window_size=4, Spatial_FFN_hidden_ratio=4, rpe=rpe).to(device)

# VPTR_Transformer = VPTR_Transformer.eval()

# #load the trained autoencoder, we initialize the discriminator from scratch, for a balanced training
# loss_dict, start_epoch = resume_training({'VPTR_Enc': VPTR_Enc, 'VPTR_Dec': VPTR_Dec}, {}, resume_AE_ckpt, loss_name_list)
# if resume_ckpt is not None:
#     loss_dict, start_epoch = resume_training({'VPTR_Transformer': VPTR_Transformer}, 
#                                              {}, resume_ckpt, loss_name_list)


# Load data 
data_set_name = 'BLOCKS' #see utils.dataset
dataset_dir = '/scratch/ms14625/VTPR/data/blocks/dataset/unlabeled'
train_loader, val_loader, test_loader, renorm_transform = get_dataloader(data_set_name, N, dataset_dir, num_past_frames, num_future_frames)
sample = next(iter(train_loader))  
past_frames, future_frames = sample
past_frames = past_frames.to(device)
future_frames = future_frames.to(device)

# Predict features 
with torch.no_grad():
    past_gt_feats = VPTR_Enc(past_frames)

    # These are features of the predicted future frame
    pred_future_feats = VPTR_Transformer(past_gt_feats)

    # These are the predicted future frames
    pred_frames = VPTR_Dec(pred_future_feats)
