import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import time 
import matplotlib.pyplot as plt
import numpy as np 

from pathlib import Path
import random
from datetime import datetime

from model import VPTREnc, VPTRDec, VPTRDisc, init_weights, VPTRFormerNAR, VPTRFormerFAR
from model import GDL, MSELoss, L1Loss, GANLoss, BiPatchNCE
from utils import KTHDataset, BAIRDataset, MovingMNISTDataset
from utils import get_dataloader
from utils import visualize_batch_clips, save_ckpt, load_ckpt, set_seed, AverageMeters, init_loss_dict, write_summary, resume_training
from utils import set_seed
from rrunet import RR_UNET 
from torchmetrics import JaccardIndex

import logging

def cal_lossD(VPTR_Disc, fake_imgs, real_imgs, lam_gan):
    pred_fake = VPTR_Disc(fake_imgs.detach().flatten(0, 1))
    loss_D_fake = gan_loss(pred_fake, False)
    # Real
    pred_real = VPTR_Disc(real_imgs.flatten(0,1))
    loss_D_real = gan_loss(pred_real, True)
    # combine loss and calculate gradients
    loss_D = (loss_D_fake + loss_D_real) * 0.5 * lam_gan

    return loss_D, loss_D_fake, loss_D_real
    
def cal_lossT(fake_imgs, real_imgs, VPTR_Disc, lam_gan):
    T_MSE_loss = mse_loss(fake_imgs, real_imgs)
    T_GDL_loss = gdl_loss(real_imgs, fake_imgs)

    if VPTR_Disc is not None:
        assert lam_gan is not None, "Please input lam_gan"
        pred_fake = VPTR_Disc(fake_imgs.flatten(0, 1))
        loss_T_gan = gan_loss(pred_fake, True)
        loss_T = T_GDL_loss + T_MSE_loss + lam_gan * loss_T_gan
    else:
        loss_T_gan = torch.zeros(1)
        loss_T = T_GDL_loss + T_MSE_loss
    
    return loss_T, T_GDL_loss, T_MSE_loss, loss_T_gan

def single_iter(VPTR_Enc, VPTR_Dec, VPTR_Disc, VPTR_Transformer, optimizer_T, optimizer_D, sample, device, lam_gan, train_flag = True):
    past_frames, future_frames = sample
    past_frames = past_frames.to(device)
    future_frames = future_frames.to(device)
    
    with torch.no_grad():
        x = torch.cat([past_frames, future_frames[:, 0:-1, ...]], dim = 1)
        gt_feats = VPTR_Enc(x)
        
    if train_flag:
        VPTR_Transformer = VPTR_Transformer.train()
        VPTR_Transformer.zero_grad(set_to_none=True)
        VPTR_Dec.zero_grad(set_to_none=True)
        
        pred_future_feats = VPTR_Transformer(gt_feats)
        pred_frames = VPTR_Dec(pred_future_feats)
        
        if optimizer_D is not None:
            assert lam_gan is not None, "Input lam_gan"
            #update discriminator
            VPTR_Disc = VPTR_Disc.train()
            for p in VPTR_Disc.parameters():
                p.requires_grad_(True)
            VPTR_Disc.zero_grad(set_to_none=True)
            loss_D, loss_D_fake, loss_D_real = cal_lossD(VPTR_Disc, pred_frames, future_frames, lam_gan)
            loss_D.backward()
            optimizer_D.step()
        
            for p in VPTR_Disc.parameters():
                    p.requires_grad_(False)

        #update Transformer (generator)
        loss_T, T_GDL_loss, T_MSE_loss, loss_T_gan = cal_lossT(pred_frames, torch.cat([past_frames[:, 1:, ...], future_frames], dim = 1), VPTR_Disc, lam_gan)
        loss_T.backward()
        nn.utils.clip_grad_norm_(VPTR_Transformer.parameters(), max_norm=max_grad_norm, norm_type=2)
        optimizer_T.step()

    else:
        if optimizer_D is not None:
            VPTR_Disc = VPTR_Disc.eval()
        VPTR_Transformer = VPTR_Transformer.eval()
        with torch.no_grad():
            pred_future_feats = VPTR_Transformer(gt_feats)
            pred_frames = VPTR_Dec(pred_future_feats)
            if optimizer_D is not None:
                loss_D, loss_D_fake, loss_D_real = cal_lossD(VPTR_Disc, pred_frames, future_frames, lam_gan)
            loss_T, T_GDL_loss, T_MSE_loss, loss_T_gan = cal_lossT(pred_frames, torch.cat([past_frames[:, 1:, ...], future_frames], dim = 1), VPTR_Disc, lam_gan)
    
    if optimizer_D is None:        
        loss_D, loss_D_fake, loss_D_real = torch.zeros(1), torch.zeros(1), torch.zeros(1)

    iter_loss_dict = {'T_total': loss_T.item(), 'T_MSE': T_MSE_loss.item(), 'T_GDL': T_GDL_loss.item(), 'T_gan': loss_T_gan.item(), 'Dtotal': loss_D.item(), 'Dfake':loss_D_fake.item(), 'Dreal':loss_D_real.item()}
    
    return iter_loss_dict

def FAR_show_sample(VPTR_Enc, VPTR_Dec, VPTR_Transformer, past_frames, num_pred=11):
    with torch.no_grad():
        past_frames = past_frames.to(device)
        past_gt_feats = VPTR_Enc(past_frames)
    
        pred_feats = VPTR_Transformer(past_gt_feats)
        for i in range(num_pred-1):
            if i == 0:
                input_feats = torch.cat([past_gt_feats, pred_feats[:, -1:, ...]], dim = 1)
            else:
                pred_future_frame = VPTR_Dec(pred_feats[:, -1:, ...])
                pred_future_feat = VPTR_Enc(pred_future_frame)
                input_feats = torch.cat([input_feats, pred_future_feat], dim = 1)

            pred_feats = VPTR_Transformer(input_feats)
    
    
        pred_frames = VPTR_Dec(pred_feats)
    
    return pred_frames[:, -1, :, :, :]

if __name__ == '__main__':
    set_seed(2021)
    ckpt_save_dir = Path('/scratch/ms14625/VTPR/VPTR_ckpts/temp')
    tensorboard_save_dir = Path('/scratch/ms14625/VTPR/VPTR_ckpts/temp_tensorboard')
    resume_AE_ckpt = '/scratch/ms14625/VTPR/VPTR_ckpts/blocks_AE_past_10_future_11_color_ckpt/epoch_6.tar'
    resume_ckpt = Path('/scratch/ms14625/VTPR/VPTR_ckpts/blocks_FAR_past_10_future_11_color_continue2_lowerlr_lowdropout_ckpt/epoch_11.tar')

    #############Set the logger#########
    if not Path(ckpt_save_dir).exists():
            Path(ckpt_save_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    format='%(asctime)s - %(message)s',
                    filename=ckpt_save_dir.joinpath('train_log.log').absolute().as_posix(),
                    filemode='a')

    start_epoch = 0
    summary_writer = SummaryWriter(tensorboard_save_dir.absolute().as_posix())
    num_past_frames = 10
    num_future_frames = 11
    encH, encW, encC = 20, 30, 528
    img_channels = 3 #3 channels for BAIR
    epochs = 20
    N = 1
    #AE_lr = 2e-4
    Transformer_lr = 1e-4
    max_grad_norm = 1.0 
    rpe = False
    lam_gan = 0.001
    dropout = 0.1
    device = torch.device('cpu')
    val_per_epochs = 1
    ngf = 128
    
    #####################Init Dataset ###########################
    data_set_name = 'BLOCKS_TEST' #see utils.dataset
    dataset_dir = '/scratch/ms14625/VTPR/data/blocks/dataset/unlabeled'
    # dataset_dir = '/scratch/ms14625/VTPR/data/blocks/dataset/unlabeled'
    test_past_frames = 10
    test_future_frames = 11
    train_loader, val_loader, test_loader, renorm_transform = get_dataloader(data_set_name, N, dataset_dir, test_past_frames, test_future_frames, bw=False)

    print(len(train_loader))
    print(len(val_loader))

    #####################Init model###########################
    VPTR_Enc = VPTREnc(img_channels, ngf=ngf, feat_dim = encC, n_downsampling = 3).to(device)
    VPTR_Dec = VPTRDec(img_channels, ngf=ngf, feat_dim = encC, n_downsampling = 3, out_layer = 'ReLU').to(device) #Sigmoid for MNIST, Tanh for KTH and BAIR
    VPTR_Enc = VPTR_Enc.eval()
    VPTR_Dec = VPTR_Dec.eval()

    VPTR_Disc = None
    # VPTR_Disc = VPTRDisc(img_channels, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d).to(device)
    # VPTR_Disc = VPTR_Disc.eval()
    # init_weights(VPTR_Disc, init_type='kaiming')
    init_weights(VPTR_Enc)
    init_weights(VPTR_Dec)

    VPTR_Transformer = VPTRFormerFAR(num_past_frames, num_future_frames, encH=encH, encW = encW, d_model=encC, 
                                nhead=8, num_encoder_layers=12, dropout=dropout, 
                                window_size=4, Spatial_FFN_hidden_ratio=4, rpe=rpe).to(device)

    optimizer_D = None
    # optimizer_D = torch.optim.Adam(params = VPTR_Disc.parameters(), lr = Transformer_lr, betas = (0.5, 0.999))
    optimizer_T = torch.optim.AdamW(params = VPTR_Transformer.parameters(), lr = Transformer_lr)

    Transformer_parameters = sum(p.numel() for p in VPTR_Transformer.parameters() if p.requires_grad)
    print(f"FAR Transformer num_parameters: {Transformer_parameters}")

    #####################Init loss function###########################
    loss_name_list = ['T_MSE', 'T_GDL', 'T_gan', 'T_total', 'Dtotal', 'Dfake', 'Dreal']
    gan_loss = GANLoss('vanilla', target_real_label=1.0, target_fake_label=0.0).to(device)
    loss_dict = init_loss_dict(loss_name_list)
    mse_loss = MSELoss()
    gdl_loss = GDL(alpha = 1)

    #load the trained autoencoder, we initialize the discriminator from scratch, for a balanced training
    loss_dict, start_epoch = resume_training({'VPTR_Enc': VPTR_Enc, 'VPTR_Dec': VPTR_Dec}, {}, resume_AE_ckpt, loss_name_list, map_location='cpu')

    start_epoch = 0 

    if resume_ckpt is not None:
        loss_dict, start_epoch = resume_training({'VPTR_Transformer': VPTR_Transformer}, 
                                                {'optimizer_T':optimizer_T}, resume_ckpt, loss_name_list)
    
        print("LOADED")
    
    VPTR_Transformer = VPTR_Transformer.eval()


    # Load segmentation model 
    seg_model = RR_UNET(3, 49).to(device)
    seg_model = torch.load("/scratch/ms14625/VTPR/rrunet_checkpoint.pt", map_location='cpu').to(device)

    # unnormalize data
    past_frames, future_frames, masks =  next(iter(test_loader))
    test_img = past_frames[0, :, :, :, :].to(device) # N, C, W, H
    orig_img = renorm_transform(test_img)

    # apply normalization in rrunet
    trans = transforms.Normalize(mean=[0.5002, 0.4976, 0.4945], std=[0.0555, 0.0547, 0.0566])  # Standard normalization
    test_img = trans(orig_img).to(device)


    # run segmentation
    # seg = seg_model(test_img.unsqueeze(0))
    # seg = torch.argmax(seg, dim=1).detach().cpu()
    # seg = np.transpose(imgs[0, :, :, :].cpu(),  (1, 2, 0))
    # orig_img = np.transpose(orig_imgs[0, :, :, :].cpu(),  (1, 2, 0))
    # axarr[0].imshow(orig_img.numpy())
    
    predicted_seg_list = []
    gt_seg_list = []
    truth_seg_list = []
    jaccard = JaccardIndex(task="multiclass", num_classes=49).to(device)
    gt_avg = 0
    predicted_avg = 0
    count = 0 

    seg_model.eval()
    with torch.no_grad():
        for past_frames, future_frames, masks in test_loader:
            masks = masks.to(device)

            # use VTPR to get last img 
            predicted_last_frame = FAR_show_sample(VPTR_Enc, VPTR_Dec, VPTR_Transformer, past_frames)
            assert len(predicted_last_frame.shape) == 4 
            predicted_last_frame_unnorm = renorm_transform(predicted_last_frame)
            
            gt_last_frame = future_frames[:, -1, :, :, :].to(device) # batch_size, C, W, H
            assert len(gt_last_frame.shape) == 4 
            gt_last_frame_unnorm = renorm_transform(gt_last_frame)

            # apply normalization in rrunet
            trans = transforms.Compose([
                transforms.Pad(padding=(0, 40), padding_mode='edge'),  # Only pad width
                # transforms.Resize((256, 256)),  # Resize the now square image to 256x256
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.5002, 0.4976, 0.4945], std=[0.0555, 0.0547, 0.0566])  # Standard normalization
            ])
            gt_last_frame = trans(gt_last_frame_unnorm).to(device)
            predicted_last_frame = trans(predicted_last_frame_unnorm).to(device)

            # Find gt frame segmentation result
            gt_seg = seg_model(gt_last_frame).to(device)
            gt_seg = torch.argmax(gt_seg, dim=1)
            gt_seg = gt_seg.unsqueeze(1)
            # gt_seg_list.append(gt_seg)
            assert len(gt_seg.shape) == 4 
            assert gt_seg.shape[1] == 1

            # gt mask
            truth_seg_list.append(masks[:, -1, :, :].unsqueeze(1))
                        
            # find predicted frame segmentation result 
            predicted_outputs = seg_model(predicted_last_frame).to(device)
            predicted_outputs = torch.argmax(predicted_outputs, dim=1)
            predicted_outputs = predicted_outputs.unsqueeze(1)
            predicted_seg_list.append(predicted_outputs)
            assert len(predicted_outputs.shape) == 4 
            assert predicted_outputs.shape[1] == 1

            import pdb; pdb.set_trace()

            # predicted_orig_img = torch.permute(predicted_last_frame_unnorm[0, :, :, :],  (1, 2, 0))
           
            # f, axarr = plt.subplots(4,1)
            # axarr[0].imshow(orig_img.numpy())    
            # axarr[1].imshow(gt_seg_list[-1].squeeze(0).cpu().numpy())
            # axarr[2].imshow(predicted_orig_img.numpy())    
            # axarr[3].imshow(predicted_seg_list[-1].squeeze(0).cpu().numpy())
            # plt.show()
            # plt.close() 


    sample_all_outputs = torch.cat(predicted_seg_list, dim=0)
    sample_all_truth = torch.cat(truth_seg_list, dim=0)
    res = jaccard(sample_all_outputs, sample_all_truth)
    print(f"res {res}", flush=True)    