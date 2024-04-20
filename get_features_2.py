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

from model import VPTREnc, VPTRDec, VPTRDisc, init_weights, VPTRFormerNAR, VPTRFormerFAR
from model import GDL, MSELoss, L1Loss, GANLoss, BiPatchNCE
from utils import KTHDataset, BAIRDataset, MovingMNISTDataset
from utils import get_dataloader
from utils import visualize_batch_clips, save_ckpt, load_ckpt, set_seed, AverageMeters, init_loss_dict, write_summary, resume_training
from utils import set_seed

import logging

if __name__ == '__main__':
    set_seed(2021)
    tensorboard_save_dir = Path('/scratch/ms14625/VTPR//VPTR_ckpts/blocks_FAR_kaiming_3_tensorboard')
    resume_AE_ckpt = '/scratch/ms14625/VTPR/VPTR_ckpts/blocks_past_10_future_11_kaiming_ckpt/epoch_28.tar'
    resume_ckpt = Path('/scratch/ms14625/VTPR/VPTR_ckpts/blocks_FAR_kaiming_ckpt/epoch_3.tar')

    #############Set the logger#########    
    start_epoch = 0
    num_past_frames = 10
    num_future_frames = 11
    encH, encW, encC = 8, 8, 128
    img_channels = 3 #3 channels for BAIR
    epochs = 3
    N = 1
    #AE_lr = 2e-4
    Transformer_lr = 1e-4
    max_grad_norm = 1.0 
    rpe = False
    lam_gan = 0.001
    dropout = 0.1
    device = torch.device('cuda')
    val_per_epochs = 1
    ngf = 64
    
    #####################Init Dataset ###########################
    data_set_name = 'BLOCKS' #see utils.dataset
    # dataset_dir = '/home/mrunal/Documents/NYUCourses/DeepLearning/project/VPTR/data/blocks/dataset/unlabeled'
    dataset_dir = '/scratch/ms14625/VTPR/data/blocks/dataset/unlabeled'
    test_past_frames = 10
    test_future_frames = 11
    train_loader, val_loader, test_loader, renorm_transform = get_dataloader(data_set_name, N, dataset_dir, test_past_frames, test_future_frames)

    print(len(train_loader))
    print(len(val_loader))

    #####################Init model###########################
    VPTR_Enc = VPTREnc(img_channels, ngf=ngf, feat_dim = encC, n_downsampling = 3).to(device)
    VPTR_Dec = VPTRDec(img_channels, ngf=ngf, feat_dim = encC, n_downsampling = 3, out_layer = 'Tanh').to(device) #Sigmoid for MNIST, Tanh for KTH and BAIR
    VPTR_Enc = VPTR_Enc.eval()
    VPTR_Dec = VPTR_Dec.eval()

    VPTR_Disc = None
    #VPTR_Disc = VPTRDisc(img_channels, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d).to(device)
    #VPTR_Disc = VPTR_Disc.eval()
    #init_weights(VPTR_Disc)
    init_weights(VPTR_Enc)
    init_weights(VPTR_Dec)

    VPTR_Transformer = VPTRFormerFAR(num_past_frames, num_future_frames, encH=20, encW = 30, d_model=encC, 
                                nhead=8, num_encoder_layers=12, dropout=dropout, 
                                window_size=4, Spatial_FFN_hidden_ratio=4, rpe=rpe).to(device)

    optimizer_D = None
    #optimizer_D = torch.optim.Adam(params = VPTR_Disc.parameters(), lr = Transformer_lr, betas = (0.5, 0.999))
    optimizer_T = torch.optim.AdamW(params = VPTR_Transformer.parameters(), lr = Transformer_lr)

    Transformer_parameters = sum(p.numel() for p in VPTR_Transformer.parameters() if p.requires_grad)
    print(f"FAR Transformer num_parameters: {Transformer_parameters}")

    #####################Init loss function###########################
    loss_name_list = ['T_MSE', 'T_GDL', 'T_gan', 'T_total', 'Dtotal', 'Dfake', 'Dreal']
    #gan_loss = GANLoss('vanilla', target_real_label=1.0, target_fake_label=0.0).to(device)
    loss_dict = init_loss_dict(loss_name_list)
    mse_loss = MSELoss()
    gdl_loss = GDL(alpha = 1)

    #load the trained autoencoder, we initialize the discriminator from scratch, for a balanced training
    loss_dict, start_epoch = resume_training({'VPTR_Enc': VPTR_Enc, 'VPTR_Dec': VPTR_Dec}, {}, resume_AE_ckpt, loss_name_list)

    start_epoch = 0 

    if resume_ckpt is not None:
        loss_dict, start_epoch = resume_training({'VPTR_Transformer': VPTR_Transformer}, 
                                                {'optimizer_T':optimizer_T}, resume_ckpt, loss_name_list)
    
        print("LOADED")
    

    sample = next(iter(train_loader))  
    past_frames, future_frames = sample
    past_frames = past_frames.to(device)
    future_frames = future_frames.to(device)

    # Predict features 
    with torch.no_grad():
        past_gt_feats = VPTR_Enc(past_frames)

        # These are features of the predicted future frame
        pred_future_feats = VPTR_Transformer(past_gt_feats)
        print(pred_future_feat.shape)