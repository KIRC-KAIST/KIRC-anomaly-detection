import os
import os.path as osp
import sys
sys.path.append('/home/test/anaconda3/envs/torch/lib/python3.7/site-packages')

import warnings
import argparse
import time
import shutil

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvideotransforms import video_transforms, volume_transforms

# Models
from Utils.utility import *


from data_loader import VideoDataset
from Utils import do_epoch




parser = argparse.ArgumentParser(description="train video anomaly detection models")

# description
parser.add_argument('--log_path', default='./log', type=str)
parser.add_argument('--exp', type=str,
                    help='setting explanation')
parser.add_argument('--gpu', required=True, type=str)

# optimization
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)
parser.add_argument('--esct', default=7, type=int)
parser.add_argument('--dim', default=16, type=int,
                   help='feature dimension')


parser.add_argument('--num_workers', default=28, type=int)

# I/O
parser.add_argument('--train', type=str, required=True)
parser.add_argument('--test', type=str, required=True)
parser.add_argument('--resize', default=224, type=int)

# model
parser.add_argument('--base_model', default='VGG19', type=str)
parser.add_argument('--pre_weight', type=str)


# techniques
parser.add_argument('--ld', type=float, default=0.05,
                   help='Lambda: hyperparameter in loss inversion')

# FGSM
parser.add_argument('--FGSM', type=str,
                   help='whether to apply FGSM attack')
parser.add_argument('--fgsm_type', type=str,
                   help='whether to apply FGSM to video or feature')
parser.add_argument('--eps', type=float, default=0.1,
                   help='epsilon for FGSM attack')



# Not in use
parser.add_argument('--fd', help='whether to apply frame difference')
parser.add_argument('--pca', type=str)
parser.add_argument('--ref_epoch', type=int, default=0,
                   help='when to apply the refurbishing')
parser.add_argument('--dr', type=float,
                    help='decay rate for noisy label')



args = parser.parse_args()


# make folders
if not os.path.exists('./log'):
    os.mkdir('./log')
    
# path for saving the model weight : write your own path
if not os.path.exists('./model_weight'):
    os.mkdir('./model_weight')

    
    
def log(message):
    with open(osp.join(args.log_path, args.exp)+'.txt', 'a+') as logger:
        logger.write(f'{message}\n')
    
def main(args):
    
    # device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    
    models = create_models(base_model=args.base_model, device=device, args=args)

    # criterion
    criterion = nn.MSELoss(reduction='none').to(device)
    
    # optimizer
    optimizers, lr_schedulers = set_optimizers(models, args)
    
    
    
        
    train_transform_list = [video_transforms.Resize((args.resize, args.resize)),
                            volume_transforms.ClipToTensor()
                           ]
    
    test_transform_list = [video_transforms.Resize((args.resize, args.resize)),
                           volume_transforms.ClipToTensor()
                           ]
    
    train_transform = video_transforms.Compose(train_transform_list) # train

    test_transform = video_transforms.Compose(test_transform_list) # test
    
    # dataset   
    df_base_path = '../data/label'
    train_df = pd.read_csv(osp.join(df_base_path, args.train+'.csv'))
    test_df = pd.read_csv(osp.join(df_base_path, args.test+'.csv'))
    
    clip_base_path = '../data/video_clip_pickles'
    train_dataset = VideoDataset(video_clip_path=osp.join(clip_base_path, args.train+'.pkl'), dataframe=train_df, frame_diff=args.fd, video_transform=train_transform)
    test_dataset = VideoDataset(video_clip_path=osp.join(clip_base_path, args.test+'.pkl'), dataframe=test_df, frame_diff=args.fd, video_transform=test_transform)

    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers, drop_last=False)
    

    
    # ------  log
    log(f"Model: {args.exp}")
    log(f"base model: {args.base_model}")
    log(f"train set : {args.train} with len: {len(train_dataset)}")
    log(f"test set : {args.test} with len: {len(test_dataset)}\n")

    log(f'Explanation: {args.exp}')
    log(f"Lambda: {args.ld}")
    log(f"Resize: {args.resize}\n")
    
    log(f'n_epochs: {args.epochs}')
    log(f'batch_size: {args.batch_size}')    
    log(f'feature dim: {args.dim}\n')
    
    log(f'num_workers: {args.num_workers}')
    log(f'Early Stop Count: {args.esct}')
    log(f"Optimizer: SGD")
    log(f"Learning Rate: {args.lr}")


    if args.FGSM:
        log('\nFGSM Attack is applied.')
        log(f"attack on -> {args.fgsm_type}")
        log(f"epsilon: {args.eps}")

    noisy = None
    
    
    
    
    
    
    
    from_=time.time()
    ## ------------- train & test
    
    #-----
    best_loss=np.inf
    best_val_recall=0.0
    best_val_auc=0.0
    best_val_auprc = 0.
    
    early_stop_count=0
    
    lr_changed=False
#     previous_lr=optimizer.state_dict()['param_groups'][0]['lr']
    feature_collector = None
    #-----
    
    for epoch in range(args.epochs):
        args.epoch = epoch+1

        log(f'\n\n##-----Epoch {args.epoch}')
        
        since = time.time()
        
        # train & validation
        args.sess = 'train'
        epoch_loss = do_epoch.Train(train_loader, models, criterion, optimizers, feature_collector, device, noisy, log, args)
        
        lr_schedulers['Spatial_Encoder'].step(epoch_loss)
        lr_schedulers['Temp_EncDec'].step(epoch_loss)
        
        
        args.sess = 'test'
        val_epoch_loss, val_total_recall, val_anomaly_recall, val_best_total_recall_threshold, val_auc, val_auprc = do_epoch.Valid(test_loader, models, criterion, device, log, args)

        
        # save_checkpoint
        
        is_best_loss=best_loss>epoch_loss
        best_loss=min(best_loss, epoch_loss)
        
        is_best_auc=best_val_auc<val_auc
        best_val_auc=max(best_val_auc, val_auc)
        
        is_best_auprc = best_val_auprc < val_auprc
        best_val_auprc = max(best_val_auprc, val_auprc)
        
        save_checkpoint({
            'epoch': epoch+1,
            'exp': args.exp,
            'state_dict': {key: models[key].state_dict() for key in models},
            'best_val_loss': best_loss,
            'best_val_recall' : best_val_recall,
            'val_best_recall_threshold':val_best_total_recall_threshold,
        }, is_best_loss, is_best_auc, is_best_auprc)
    
        end=time.time()
                
        if is_best_loss:
            log('\n---- Best Train Loss ----')
            
        if is_best_auc:
            log('\n---- Best Val AUC')
            
        if is_best_auprc:
            log('\n---- Best Val AUPRC')
            
        log(f'\nRunning Time: {int((end-since)//60)}m {int((end-since)%60)}s\n')
        

        # early stopping
        if is_best_auc or is_best_auprc:
            early_stop_count=0
        else:
            early_stop_count+=1

        log(f'Early_stop_count: {early_stop_count}\n\n')

        if early_stop_count==args.esct:
            log(f'\nEarly Stopped because Validation AUC or AUPRC is not increasing for {args.esct} epochs')
            break
            
          

    to_=time.time()
    log(f'\nTotal Running Time: {int((to_-from_)//60)}m {int((to_-from_)%60)}s')
    

    
    
    
    
    
def save_checkpoint(state, is_best_loss, is_best_auc, is_best_auprc, filename='./checkpoint/'+args.exp+'.pth'):    

    if is_best_auc:
        torch.save(state, './model_weight/'+args.exp+'_best_AUC.pth')
                
    if is_best_auprc:
        torch.save(state, './model_weight/'+args.exp+'_best_AUPRC.pth')

    
if __name__=='__main__':
    main(args)
    