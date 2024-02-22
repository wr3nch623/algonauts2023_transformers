# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from scipy.stats import pearsonr as corr

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test
from models import build_model

from utils import *
#from datasets.loaddata_g import *
from dataset_algonauts import fetch_dataloader

# import wandb
# os.environ['WANDB_MODE'] = 'offline'

import code
import pprint


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    
    
    ## algonauts params
    parser.add_argument('--subj', default=1, type=int)  # 5 is a good test sub
    parser.add_argument('--run', default=1, type=int)  # 5 is a good test sub
    parser.add_argument('--data_dir', default='~/AI/algonauts_2023_challenge_data/', type=str)
    parser.add_argument('--parent_submission_dir', default='../algonauts_2023_challenge_submission/', type=str)
    
    parser.add_argument('--saved_feats', default=None, type=str) #'dinov2q'
    parser.add_argument('--saved_feats_dir', default='../../algonauts_image_features/', type=str) 
    
    parser.add_argument('--decoder_arch', default='transformer', type=str) #'dinov2q'
    parser.add_argument('--readout_res', default='streams_inc', type=str)   
    # [streams_inc, visuals, bodies, faces, places, words, hemis]

    # Model parameters
    
    #'../results/detr_grouping_256_2/checkpoint_0.50300_1.pth'
    parser.add_argument('--resume', default=None, help='resume from checkpoint')

    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    
    parser.add_argument('--pretrained', type=str, default=None,
                        help="all the weights that can be, will be initialized from this pretrained model") 
                            #'../../pretrained/detr_r50.pth'
    
#     parser.add_argument('--pretrained_params', type=str, default='backbone input_projection encoder decoder',
#                         help="To limit the scope of what can be initialized using the pretrained model") 
    
    parser.add_argument('--frozen_params', type=str, default='backbone input_proj',
                        help="These components will not be retrained")    
    
    # * Backbone
    parser.add_argument('--backbone', default='dinov2', type=str,
                        help="Name of the convolutional backbone to use")  #resnet50
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=0, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=1, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=768, type=int,
                        help="Size of the embeddings (dimension of the transformer)")  #256  #868 (100+768) 
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=16, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=16, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    
    parser.add_argument('--enc_output_layer', default=1, type=int,
                    help="Specify the encoder layer that provides the encoder output. If None, will be the last layer")
    
    parser.add_argument('--output_layer', default='backbone', type=str,
                    help="If no encoder (enc_layers = 0), what to use to feed to the linear classifiers; input_proj or backbone")

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_true',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=1, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    
    
    # other params
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--class_loss_coef', default=1, type=float)
    parser.add_argument('--cosine_loss_coef', default=0, type=float)
    
    parser.add_argument('--task', default = 'algonauts') 
    parser.add_argument('--distributed', default=0, type=int)
    
    parser.add_argument('--wandb', default='subj03_run3', type=str)

    # dataset parameters
    parser.add_argument('--dataset_grouping_dir', default='./datasets/dataset_grouping/')
    
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default ='../../coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./results/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument('--save_model', default=1, type=int) 

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    # utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))
    
    print('start')

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    

#     args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#     print(args.device)
    device = torch.device(args.device)

    # fix the seed for reproducibility
#     seed = args.seed + utils.get_rank()
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
    
    args.subj = format(args.subj, '02')
    args.data_dir = os.path.join(args.data_dir, 'subj'+args.subj)
    args.subject_submission_dir = os.path.join(args.parent_submission_dir,
        'subj'+args.subj)
    
    save_dir = args.output_dir + 'detr_dino_' + str(args.enc_output_layer) + '_' + args.readout_res + '_' + str(args.num_queries) + '/' + str(args.subj) + '/run' + str(args.run) + '/' 
    args.save_dir = save_dir
    #print(save_dir)
    #print(device)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create the submission directory if not existing
    if not os.path.isdir(args.subject_submission_dir):
        os.makedirs(args.subject_submission_dir)

            
    # Load the ROI classes mapping dictionaries
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
        'mapping_floc-faces.npy', 'mapping_floc-places.npy',
        'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(args.data_dir, 'roi_masks', r),
            allow_pickle=True).item())

    # Load the ROI brain surface maps
    lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
        'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
        'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
        'lh.streams_challenge_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
        'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
        'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
        'rh.streams_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []
    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(np.load(os.path.join(args.data_dir, 'roi_masks',
            lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(args.data_dir, 'roi_masks',
            rh_challenge_roi_files[r])))
        

    print(lh_challenge_rois)
    #print('complete')



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)