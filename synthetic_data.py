
import math
import pandas as pd
import torch
import numpy as np
from torch import nn
from torchdiffeq import odeint
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#torch.manual_seed(42)

from data_utils import SSA
from models import massice
from plotting import plotsynthetic, plotsynoverview

def getsyntheticdata(traindata,nonoise=False,saveplots=False):
    #traindata = torch.load('Data/LevData.pth')

    nexps = len(traindata)
    #print(nexps)
    maxexplength = 1500

    # exact values
    RHOICE = 910.0

    massratio = traindata.massratio
    Temp = traindata.T.unsqueeze(dim=1).to(DEVICE)
    Si = traindata.Si.unsqueeze(dim=1).to(DEVICE)
    P = traindata.P.unsqueeze(dim=1).to(DEVICE)

    r0 = traindata.r0

    m0 = 4 / 3 * math.pi * RHOICE * r0 ** 3
    m0 = m0.unsqueeze(dim=1).to(DEVICE)
    time = torch.arange(0, maxexplength, 1).float()

    #print(m0.shape)
    model_nelson = massice(depmodel="nelson").float().to(DEVICE)
    model_nosurfk = massice(depmodel="constant", c=1.0).float().to(DEVICE)

    mass_nelson = model_nelson(m0.to(DEVICE), time.squeeze(dim=0).to(DEVICE), Temp, Si).detach()
    mass_nosurfk = model_nosurfk(m0.to(DEVICE), time.squeeze(dim=0).to(DEVICE), Temp, Si).detach()

    m0 = m0.unsqueeze(dim=2).expand(mass_nelson.shape)

    # synthetic data sets
    predmasses_nelson = (mass_nelson / m0)[:, :, 0]
    predmasses_nosurfk = (mass_nosurfk / m0)[:, :, 0]

    # real data sets
    massratios = massratio.cpu().detach()

    times = time.cpu().detach()

    # Use SSA to detrend noise and add to synthetic signals
    massratiostrend = np.zeros((massratios.shape[0], maxexplength))
    massratiosnoise = np.zeros((massratios.shape[0], maxexplength))
    ssatime = np.zeros(massratios.shape[0])
    window_size = 60
    import time

    if nonoise==True:
        predmassesReal = predmasses_nelson
    else:
        print("Adding detrended noise from real experiments to synthetic data")
        for i in tqdm(range(0, massratios.shape[0])):
            start = time.time()
            n = traindata.explength[i].item()
            # print(i,n)
            n = np.min((n, maxexplength))  # n<25000:
            ssa0 = SSA(pd.Series(massratios[i, :n]), window_size)
            end = time.time()
            ssatime[i] = end - start

            massratiostrend[i, :n] = ssa0.reconstruct([0, 1])
            massratiosnoise[i, :n] = ssa0.reconstruct(np.arange(2, 60, 1).tolist())

        predmassesReal = predmasses_nelson+massratiosnoise
        if saveplots==True:
            plotsynthetic(traindata,predmassesReal,predmasses_nelson,massratiosnoise)

    alpha_nelson = model_nelson.dmidt.alphas(Temp, Si)
    plotsynoverview(Temp, Si, alpha_nelson, predmassesReal)

    return predmassesReal,alpha_nelson