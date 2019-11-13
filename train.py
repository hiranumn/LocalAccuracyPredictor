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
    model = pyErrorPred.Model(obt_size=70,
                              tbt_size=33,
                              prot_size=None,
                              num_chunks=5,
                              optimizer="adam",
                              mask_weight=0.33,
                              lddt_weight=10.0,
                              feature_mask = masks,
                              ignore3dconv = args.no_3dconv,
                              name=args.folder)
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
        