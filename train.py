import sys
import argparse
import os
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import pandas as pd
import multiprocessing

import sys
sys.path.insert(0, "./")
import pyErrorPred

def main():
    parser = argparse.ArgumentParser(description="Error predictor network trainer",
                                     epilog="v0.0.1")
    
    parser.add_argument("folder",
                        action="store",
                        help="Location of folder to save checkpoints to.")
    
    parser.add_argument("--epoch",
                        "-e", action="store",
                        type=int,
                        default=200,
                        help="# of epochs (path over all proteins) to train for (Default: 200)")
    
    parser.add_argument("--no_rosetta",
                        "-nros",
                        action="store_true",
                        default=False,
                        help="Training without Rosetta energy features (Default: False)")
    
    parser.add_argument("--no_orientation",
                        "-nori",
                        action="store_true",
                        default=False,
                        help="Training without orientation features (Default: False)")
    
    parser.add_argument("--no_secondary_struct",
                        "-nss",
                        action="store_true",
                        default=False,
                        help="Training without secondary structure features (Default: False)")
    
    parser.add_argument("--no_amino_acid",
                        "-naa",
                        action="store_true",
                        default=False,
                        help="Training without amino acid features (Default: False)")
    
    parser.add_argument("--no_angles",
                        "-nang",
                        action="store_true",
                        default=False,
                        help="Training without backbone geometry features (Default: False)")
    
    parser.add_argument("--no_3dconv",
                        "-n3d",
                        action="store_true",
                        default=False,
                        help="Training without 3d convolutions (Default: False)")
    
    parser.add_argument("--no_cutoff",
                        "-ncut",
                        action="store_true",
                        default=False,
                        help="Not having the -6 cutoff for auxiliary distance matrix (Default: False)")
    
    parser.add_argument("--esto_loss_only",
                        "-elo",
                        action="store_true",
                        default=False,
                        help="Estogram loss only (Default: False)")
    
    parser.add_argument("--no_last_dilation",
                        "-nld",
                        action="store_true",
                        default=False,
                        help="not using dilation for very last part of resnet (Default: False)")
    
    parser.add_argument("--scaled_loss",
                        "-scl",
                        action="store_true",
                        default=False,
                        help="using loss scaled with protein size. (Default: False)")
    
    parser.add_argument("--label_smoothing",
                        "-lsm",
                        action="store_true",
                        default=False,
                        help="apply label smoothing at training time. (Default: False)")
    
    parser.add_argument("--new",
                        "-new",
                        action="store_true",
                        default=False,
                        help="apply nrestype 20 instead of 167. Should be used for all newer architectures (Default: False)")
    
    parser.add_argument("--partial",
                        "-partial",
                        action="store_true",
                        default=False,
                        help="partial label smoothing. (Default: False)")
    
    parser.add_argument("--transpose_matrix",
                        "-transmtx",
                        action="store_true",
                        default=False,
                        help="Transpose last few blocks (Default: False)")

    parser.add_argument("--self_attention",
                        "-selfattn",
                        action="store_true",
                        default=False,
                        help="Put self attention on last few blocks(Default: False)")

    parser.add_argument("--decay",
                        "-d", action="store",
                        type=float,
                        default=0.99,
                        help="Decay rate for learning rate (Default: 0.99)")
    
    parser.add_argument("--base",
                        "-b", action="store",
                        type=float,
                        default=0.0005,
                        help="Base learning rate (Default: 0.0005)")
    
    parser.add_argument("--silent",
                        "-s",
                        action="store_true",
                        default=False,
                        help="Run in silent mode (Default: False)")
   
    args = parser.parse_args()
    
    restoreModel = False
    if isdir(args.folder):
        restoreModel = True
    
    #########################
    ### Generating a mask ###
    #########################
    ignore_list = []
    if args.no_rosetta: ignore_list.append("rosetta")
    if args.no_orientation: ignore_list.append("orientation")
    if args.no_secondary_struct: ignore_list.append("ss")
    if args.no_amino_acid: ignore_list.append("aa")
    if args.no_angles: ignore_list.append("angles")
    masks = pyErrorPred.getMask(ignore_list)
    if not args.silent:
        print("1 dimensinal features:", 70-len(masks[0]), "of 70")
        print("2 dimensinal features:", 33-len(masks[1]), "of 33")
        print("Ignoring 3d convolution:", args.no_3dconv)
        
    if not args.silent:
        print("Loading samples")
    ##########################
    ### Loading data files ###
    ##########################
    script_dir = os.path.dirname(__file__)
    base = join(script_dir, "data/")
    if args.no_cutoff:
        X = pyErrorPred.dataloader(np.load(join(base,"train_proteins.npy")),
                                   lengthmax=280,
                                   distribution=False,
                                   distance_cutoff=0)
        V = pyErrorPred.dataloader(np.load(join(base,"valid_proteins.npy")),
                                   lengthmax=280,
                                   distribution=False,
                                   distance_cutoff=0)
    else:
        X = pyErrorPred.dataloader(np.load(join(base,"train_proteins.npy")),
                                   lengthmax=280,
                                   distribution=False)
        V = pyErrorPred.dataloader(np.load(join(base,"valid_proteins.npy")),
                                   lengthmax=280,
                                   distribution=False)
    
    if not args.silent:
        print("Building a network")
    #########################
    ### Training a model  ###
    #########################
    if not args.esto_loss_only:
        model = pyErrorPred.Model(obt_size=70,
                                  tbt_size=33,
                                  prot_size=None,
                                  num_chunks=5,
                                  optimizer="adam",
                                  mask_weight=0.33,
                                  lddt_weight=10.0,
                                  feature_mask = masks,
                                  ignore3dconv = args.no_3dconv,
                                  name=args.folder,
                                  scaled_loss=args.scaled_loss,
                                  label_smoothing=args.label_smoothing,
                                  no_last_dilation = args.no_last_dilation,
                                  partial_instance_norm = args.partial,
                                  transpose_matrix = args.transpose_matrix,
                                  self_attention = args.self_attention,
                                  nretype = 20 if args.new else 167)
    else:
        model = pyErrorPred.Model(obt_size=70,
                                  tbt_size=33,
                                  prot_size=None,
                                  num_chunks=5,
                                  optimizer="adam",
                                  mask_weight=0.01,
                                  lddt_weight=0.01,
                                  feature_mask = masks,
                                  ignore3dconv = args.no_3dconv,
                                  name=args.folder,
                                  scaled_loss=args.scaled_loss,
                                  label_smoothing=args.label_smoothing,
                                  no_last_dilation = args.no_last_dilation, 
                                  partial_instance_norm = args.partial,
                                  transpose_matrix = args.transpose_matrix,
                                  self_attention = args.self_attention,
                                  nretype = 20 if args.new else 167)
    if restoreModel:
        model.load()
   
    if not args.silent:
        print("Training the network")
    model.train(X,
                V,
                args.epoch,
                decay=args.decay,
                base_learning_rate=args.base,
                save_best=True,
                save_freq=10)
    
    return 0

if __name__== "__main__":
    main()
        