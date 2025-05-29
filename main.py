from evaluate import evaluate, fit_alpha,fit_gc

import functools
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.optim as Optimizer
from torchdiffeq import odeint, odeint_event #https://github.com/rtqichen/torchdiffeq

import torch.utils.data as thdat

from typing import Sequence, Optional,Union, List
from tqdm import tqdm

import glob
import os
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#torch.manual_seed(42)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from data_utils import levdiffdata, SSA, getname,SubsetWithAttrs
from synthetic_data import getsyntheticdata
from models import massice
from train import trainmodels
from model_comparison import evaluate_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Loop")

    parser.add_argument("-s", "--synthetic", action='store_true', help="use synthetic data (default = True)")
    parser.add_argument("-nn", "--nonoise", action='store_true', help="don't add noise to synthetic data")
    parser.add_argument('-sw', "--strong", action='store_true', help="train strong (weak) model (default = True)")
    parser.add_argument("-n", "--num_iterations", default=1000, type=int, help="number of training epochs (default = 1000)")
    parser.add_argument("-lr", "--base_lr", default=0.01, type=float, help="initial learning rate (default = 0.01)")
    parser.add_argument("-d", "--decay", default=0.95, type=float, help="decay rate for cosine decay (default = 0.95)")
    parser.add_argument("--maxexplength", default = 500, type = int, help="maximum experiment length (default = 500)")
    parser.add_argument("-l","--load", action='store_true', help = "load model checkpoint")
    parser.add_argument("-lsr", "--loadSR", action='store_true', help="load PySR checkpoint")
    parser.add_argument("--saveplots", action='store_true', help="Save plots")
    parser.add_argument("--randomseed",default=42,help="random seed (default = 42)")
    parser.add_argument("--exclude",action='store_true', help = "exclude bad experiments in training data")
    parser.add_argument("--allpoints",action='store_true', help = "include every 50th point when fitting SR for weak case")

    args = parser.parse_args()

    torch.manual_seed(args.randomseed)
    np.random.seed(args.randomseed)

    traindata = torch.load('Data/LevDataUncertainty.pth')

    # exclude data sets with obvious inconsistencies
    if args.exclude:
        badexp = [20, 52, 171, 278, 279, 282, 283]
        marginalexp = [6, 8, 15, 42, 45, 55, 61, 65, 75, 79, 88, 115, 117, 120, 123, 125, 144, 145, 152, 164, 175, 185, 192,
                       222, 226, 227]
        all_indices = set(range(len(traindata)))
        include_indices = sorted(list(all_indices - set(badexp + marginalexp)))
        traindata = SubsetWithAttrs(traindata, include_indices)

    name = getname(args)
    print("# Experiments: ",len(traindata))

    # use synthetic data that assumes the Nelson and Baker parameterization
    if args.synthetic:
        print("Using synthetic data")
        massratio,alphas = getsyntheticdata(traindata,nonoise=args.nonoise,saveplots=args.saveplots)
        massratio = massratio.detach().float()
    else:
        massratio = traindata.massratio.float()

    # load a pretrained model
    if args.load:
        #checkpointpath = "/Users/karalamb/Columbia/Projects/DepositionalIce/IceNODE/Checkpoints/"
        checkpointname = "Checkpoints/Checkpoint_" + name + ".pt"
        checkpoint = torch.load(checkpointname)
        print("Loading "+checkpointname)
        if args.strong == True:  # strong form of the model
            model = massice(depmodel="NN").float().to(DEVICE)
        else:
            model = massice(strong=False).float().to(DEVICE)
        #model.load_state_dict(checkpoint['model_state_dict'])
        dirpath = "/Users/karalamb/Columbia/Projects/DepositionalIce/IceSciML"
        model.load_state_dict(torch.load(os.path.join(dirpath,"Real_weakNODE500_L2unscaled_mscaled_noharrison.pt")))
    else:
        print("Start training")
        model = trainmodels(args,traindata,massratio)

    # evaluate the model - move this after PySR fits and evaluate the fit expressions too.
    print("Evaluating the model")
    df = evaluate(args,traindata,massratio,model)

    # fit symbolic expressions
    print("Fitting symbolic expression")
    if args.strong == True:
        pysrmodule = fit_alpha(df,name,loadfile=args.loadSR)
    else:
        if args.allpoints:
            name = name+"_allpoints"
        pysrmodule = fit_gc(df,name,loadfile=args.loadSR)

    # print off interpolation and extrapolation results for models (Table 1)
    print("Comparing models")
    nsr = len(pysrmodule.equations_)

    for i in range(1,nsr): # ignore the 1st in case it is a constant.
        print(i,pysrmodule.equations_['score'][i],pysrmodule.equations_['loss'][i],pysrmodule.equations_['complexity'][i])
        print(pysrmodule.sympy(i))

        learnedfunction = pysrmodule.pytorch(i)
        evaluate_models(args,traindata,massratio,model,learnedfunction)

    # comparison with AIDA experiments


