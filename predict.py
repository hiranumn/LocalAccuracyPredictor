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
    parser = argparse.ArgumentParser(description="Error predictor network prediction generator",
                                     epilog="v0.0.1")
    
    parser.add_argument("name",
                        action="store",
                        help="Modelname")
    
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
    
    parser.add_argument("--silent",
                        "-s",
                        action="store_true",
                        default=False,
                        help="Run in silent mode (Default: False)")
   
    args = parser.parse_args()
    
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
        
    model_dir = "/net/scratch/hiranumn/models"
    
    model_suffix = args.name
    model_names = [i for i in listdir(model_dir) if model_suffix in i]
    ##########################
    ### Loading data files ###
    ##########################
    #########################
    ### Training a model  ###
    #########################
    for model_index in range(len(model_names)):
        if not args.silent:
            print("Building a network:", join(model_dir, model_names[model_index]))
        model = pyErrorPred.Model(obt_size=70,
                                  tbt_size=33,
                                  prot_size=None,
                                  num_chunks=5,
                                  optimizer="adam",
                                  mask_weight=0.33,
                                  lddt_weight=10.0,
                                  feature_mask = masks,
                                  ignore3dconv = args.no_3dconv,
                                  name=join(model_dir, model_names[model_index]))
        model.load()
   

        try:
            os.mkdir(join("/net/scratch/hiranumn/preds/", model_names[model_index]))
        except:
            pass
        out_base = join("/net/scratch/hiranumn/preds/", model_names[model_index])
        script_dir = os.path.dirname(__file__)
        base = join(script_dir, "data/")
        test = np.load(join(base, "test_proteins.npy"))
        for t in test:
            X = pyErrorPred.dataloader([t],
                                   lengthmax=280,
                                   distribution=False)
            if t in X.samples_dict:
                if not args.silent:
                    print("Working on", t)
                try:
                    os.mkdir(join(out_base, t))
                except:
                    pass
                out_dir = join(out_base, t)
                n = len(X.samples_dict[t])
                for i in range(n):
                    name = X.samples_dict[t][i]
                    if name != "native":
                        output = {}
                        batch = X.next(i)
                        output = model.predict(batch)
                        np.savez_compressed(join(out_dir, name+".npz"),
                                            lddt      = output[0][0],
                                            esto      = output[0][1],
                                            mask      = output[0][2],
                                            lddt_true = output[1][0],
                                            esto_true = output[1][1],
                                            mask_true = output[1][2])

    return 0

if __name__== "__main__":
    main()
        