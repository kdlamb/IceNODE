import pysr
from pysr.expression_specs import *

import functools
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

import torch
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

from plotting import compare_mratios_grid,plot_alpha_fits, plot_Gc_fits,plot_Gc_functionaldependence
from models import massice
from data_utils import getname


def evaluate_models(args, traindata, massratio_real, model, pysrmodule, comparisonmodel=None, comparisonname=None):
    #
    nexps = len(traindata)
    maxexplength = args.maxexplength
    loss1 = nn.MSELoss(reduction="none")

    massratio = massratio_real.float().to(DEVICE)
    Temp = traindata.T.unsqueeze(dim=1).float().to(DEVICE)
    Si = traindata.Si.unsqueeze(dim=1).float().to(DEVICE)
    P = traindata.P.unsqueeze(dim=1).float().to(DEVICE)
    r0 = traindata.r0

    RHOICE = 910.0
    m0 = 4 / 3 * math.pi * RHOICE * r0 ** 3
    m0 = m0.unsqueeze(dim=1)

    model_nelson = massice(depmodel="nelson").float().to(DEVICE)
    model_nosurfk = massice(depmodel="constant", c=1.0).float().to(DEVICE)

    #calculate models at different lengths
    evallengths = [500,1000,1500]
    nlengths = len(evallengths)

    nmodels = 4
    mselosses = np.zeros((nlengths,nmodels))
    lossesbyexps = np.zeros((nexps,nlengths,nmodels))

    # pysr
    learnedfunction = pysrmodule.pytorch()

    for i in range(0,nlengths):
        evallength = evallengths[i]
        time = torch.arange(0, evallength, 1).float()
        lengths = traindata.explength.clone()
        lengths[np.where(lengths > evallength)] = evallength

        lossmask = torch.zeros((nexps, evallength))
        for k in range(0, nexps):
            lossmask[k, 0:lengths[k]] = 1

        mass_NODE = model(m0.to(DEVICE), time.squeeze(dim=0).to(DEVICE), Temp, Si).detach()
        mass_nelson = model_nelson(m0.to(DEVICE), time.squeeze(dim=0).to(DEVICE), Temp, Si).detach()
        mass_nosurfk = model_nosurfk(m0.to(DEVICE), time.squeeze(dim=0).to(DEVICE), Temp, Si).detach()

        m0s = m0.unsqueeze(dim=2).expand(mass_NODE.shape)
        massratios_NODE = (mass_NODE.cpu().detach() / m0s)[:, :, 0]
        massratios_nelson = (mass_nelson.cpu().detach() / m0s)[:, :, 0]
        massratios_nosurfk = (mass_nosurfk.cpu().detach() / m0s)[:, :, 0]

        massratios = [massratios_NODE, massratios_nelson, massratios_nosurfk]
        modelnames = ["Current","Nelson","No Surf. K."]

        # Integrating model with symbolic regression expression
        if args.strong == True:
            model_SR = massice(depmodel="SR",learnedfunction=learnedfunction).float().to(DEVICE)
            mass_SR = model_SR(m0.to(DEVICE), time.squeeze(dim=0).to(DEVICE), Temp, Si).detach()
            m0s = m0.unsqueeze(dim=2).expand(mass_SR.shape)
            massratios_SR = (mass_SR.cpu().detach() / m0s)[:, :, 0]

            massratios.append(massratios_SR)
            modelnames.append("Best SR")
        else:
            model_SR = massice(gcmodel="SR",learnedfunction=learnedfunction).float().to(DEVICE)
            mass_SR = model_SR(m0.to(DEVICE), time.squeeze(dim=0).to(DEVICE), Temp, Si).detach()
            m0s = m0.unsqueeze(dim=2).expand(mass_SR.shape)
            massratios_SR = (mass_SR.cpu().detach() / m0s)[:, :, 0]

            massratios.append(massratios_SR)
            modelnames.append("Best SR")

        # Other models to compare with
        if comparisonmodel is not None:
            if comparisonname is not None:
                pass
            else:
                print("Need names of comparison models to compare")
            pass

        for j in range(len(massratios)):
            lossbylength = loss1(massratios[j][:,0:evallength].float(), massratio_real[:,0:evallength].float()) * lossmask
            lossbyexp = torch.sum(lossbylength, dim=1)
            lossODE = torch.sum(lossbyexp)
            mselosses[i,j] = lossODE.item()
            lossesbyexps[:,i,j] = lossbyexp.detach().numpy()

    minloss = np.argmin(lossesbyexps[:,0,:], axis=1)
    unique_values, counts = np.unique(minloss, return_counts=True)
    bestexps = np.zeros(nmodels)
    bestexps[unique_values] = counts

    df = pd.DataFrame(mselosses.T, index=modelnames, columns=evallengths)
    df["# Best (500)"] = bestexps
    df["% Best (500)"] = 100*bestexps/nexps
    pd.options.display.float_format = '{:.0f}'.format
    print(df)



