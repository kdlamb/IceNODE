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

from data_utils import levdiffdata, SSA, getname
from synthetic_data import getsyntheticdata
from models import massice
from constants import constants

def cosine_decay(learning_rate, global_step, decay_steps, alpha=0.0):
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return learning_rate * decayed
def learning_rate_schedule(global_step, warmup_steps, base_learning_rate, lr_scaling, train_steps):
    warmup_steps = int(round(warmup_steps))
    scaled_lr = base_learning_rate * lr_scaling
    if warmup_steps:
        learning_rate = global_step / warmup_steps * scaled_lr
    else:
        learning_rate = scaled_lr

    if global_step < warmup_steps:
        learning_rate = learning_rate
    else:
        learning_rate = cosine_decay(
            scaled_lr, global_step - warmup_steps, train_steps - warmup_steps
        )
    return learning_rate
def set_learning_rate(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr
def trainmodels(args,traindata,massratio):
    # values for training
    base_lr = args.base_lr
    num_iterations = args.num_iterations
    decay = args.decay
    nexps = len(traindata)
    maxexplength = args.maxexplength
    loss1 = nn.MSELoss(reduction="none")

    massratio = massratio.float().to(DEVICE)
    Temp = traindata.T.unsqueeze(dim=1).float().to(DEVICE)
    Si = traindata.Si.unsqueeze(dim=1).float().to(DEVICE)
    P = traindata.P.unsqueeze(dim=1).float().to(DEVICE)
    r0 = traindata.r0

    m0 = 4 / 3 * math.pi * constants.RHOICE * r0 ** 3
    m0 = m0.unsqueeze(dim=1)
    time = torch.arange(0, maxexplength, 1).float()
    #print(maxexplength)

    lengths = traindata.explength.clone()
    lengths[np.where(lengths > maxexplength)] = maxexplength

    lossmask = torch.zeros((nexps, maxexplength))

    for i in range(0, nexps):
        lossmask[i, 0:lengths[i]] = 1

    # loss
    losses = []
    bestfits = np.zeros((nexps, num_iterations, maxexplength))

    if args.physics == "strong": # strong form of the model
        print("Using strong physics constraint")
        model = massice(physics=args.physics,depmodel="NN").float().to(DEVICE)
    elif args.physics == "medium": # fit dterm
        print("Using medium physics constraint")
        model = massice(physics=args.physics,dtermmodel="NN").float().to(DEVICE)
    else: # weak - fit G transfer coefficient
        print("Using weak physics constraint")
        model = massice(physics=args.physics,gmodel="NN").float().to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr)

    model.train()

    for itr in tqdm(range(num_iterations)):
        optimizer.zero_grad()

        mass = model(m0.float().to(DEVICE), time.squeeze(dim=0).to(DEVICE), Temp, Si)

        morig = m0.unsqueeze(dim=2).expand(mass.shape)
        predmasses = (mass / morig)[:, :, 0]

        lossbylength = loss1(predmasses[:, 0:maxexplength].float(), massratio[:, 0:maxexplength].float()) * lossmask
        lossODE = torch.sum(lossbylength, dim=1)

        lossODE = torch.sum(lossODE)

        loss = lossODE
        bestfits[:, itr, :] = (mass.cpu().detach() / morig)[:, :, 0]

        loss.backward()

        lr = learning_rate_schedule(itr, 0, base_lr, 1.0, num_iterations)
        set_learning_rate(optimizer, lr)
        optimizer.step()

        losses.append(loss.item())

        if itr%(int(args.num_iterations/10)) == 0:
            print(itr, loss.item(), lr)

    # if args.synthetic:
    #     synreal = "Synthetic"
    # else:
    #     synreal = "Real"
    # if args.strong:
    #     sw = "Strong"
    # else:
    #     sw = "Weak"
    #
    # #name = "{}_{}_{:04d}".format(synreal, sw,args.num_iterations)
    # #print(name)
    name = getname(args)
    print(name)

    lossfigname = "Figures/Losses_"+name+".png"
    checkpointname = "Checkpoints/Checkpoint_"+name+".pt"

    plt.plot(losses, label="Total loss")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig(lossfigname)
    plt.close()

    checkpoint = {
        'num_iterations': args.num_iterations,
        'base_lr': base_lr,
        'decay': decay,
        'randomseed': args.randomseed,
        'maxexplength': maxexplength,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss':losses}

    torch.save(checkpoint, checkpointname)

    return model