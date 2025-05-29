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


def evaluate(args,traindata,massratio,model):
    massratio_real = massratio.float().to(DEVICE)
    Temp = traindata.T.unsqueeze(dim=1).float().to(DEVICE)
    Si = traindata.Si.unsqueeze(dim=1).float().to(DEVICE)
    P = traindata.P.unsqueeze(dim=1).float().to(DEVICE)
    nucleation = traindata.nucleation.unsqueeze(dim=1).float().to(DEVICE)
    r0 = traindata.r0
    explengths = traindata.explength

    RHOICE = 910.0
    m0 = 4 / 3 * math.pi * RHOICE * r0 ** 3
    m0 = m0.unsqueeze(dim=1)
    time = torch.arange(0, args.maxexplength, 1).float()

    model_nelson = massice(depmodel="nelson").float().to(DEVICE)
    model_nosurfk = massice(depmodel="constant", c=1.0).float().to(DEVICE)

    mass_NODE = model(m0.to(DEVICE), time.squeeze(dim=0).to(DEVICE), Temp, Si).detach()
    mass_nelson = model_nelson(m0.to(DEVICE), time.squeeze(dim=0).to(DEVICE), Temp, Si).detach()
    mass_nosurfk = model_nosurfk(m0.to(DEVICE), time.squeeze(dim=0).to(DEVICE), Temp, Si).detach()

    m0 = m0.unsqueeze(dim=2).expand(mass_NODE.shape)
    massratios_NODE = (mass_NODE.cpu().detach() / m0)[:, :, 0]
    massratios_nelson = (mass_nelson.cpu().detach() / m0)[:, :, 0]
    massratios_nosurfk = (mass_nosurfk.cpu().detach() / m0)[:, :, 0]

    massratios = [massratio_real,massratios_NODE, massratios_nelson, massratios_nosurfk]


    if args.synthetic:
        synreal = "Synthetic"
    else:
        synreal = "Real"
    if args.strong:
        sw = "Strong"
    else:
        sw = "Weak"

    names = [synreal,"NODE", "nelson", "nosurfk"]

    #plotname = "Fits_{}_{}_{:04d}".format(synreal, sw,args.num_iterations)
    plotname = getname(args)
    compare_mratios_grid(massratios, explengths,names, plotname, ngrids=5)

    if args.strong == True:
        alphas_pred = model.dmidt.alphas(Temp, Si)
        alphas_nelson = model_nelson.dmidt.alphas(Temp, Si)
        d = {"Temp": Temp.detach().squeeze(dim=1).numpy(),
             "Si": Si.detach().squeeze(dim=1).numpy(),
             "Alpha_pred": alphas_pred.detach().squeeze(dim=1).numpy(),
             "Alpha_Nelson": alphas_nelson.detach().squeeze(dim=1).numpy()}
        df = pd.DataFrame(data=d)
        df.to_csv("Fits/"+plotname+"_fits.csv")
    else:
        allm = args.allpoints # uses more than just the first few points
        if allm:
            Tempall = torch.broadcast_to(Temp[:, None, :], (290, 500, 1)).reshape(290 * 500, 1)
            Siall = torch.broadcast_to(Si[:, None, :], (290, 500, 1)).reshape(290 * 500, 1)
            massall = (m0[:, :, :] * massratio[:, :500].unsqueeze(dim=2)).reshape(290 * 500, 1)

            deff, dsph = model.dmidt.getdeff(massall[::50].to(DEVICE),Tempall[::50], Siall[::50])

            d = {"Mass": massall[::50].detach().squeeze(dim=1).numpy(),
                 "Temp": Tempall[::50].squeeze(dim=1).numpy(),
                 "Si": Siall[::50].squeeze(dim=1).numpy(),
                 "Deff": deff.detach().squeeze(dim=1).numpy(),
                 "Dsph": dsph.detach().squeeze(dim=1).numpy()}
        else:
            deff, dsph = model.dmidt.getdeff(m0[:,0,:].to(DEVICE),Temp, Si)

            d = {"Mass": m0[:,0,:].detach().squeeze(dim=1).numpy(),
                 "Temp": Temp.squeeze(dim=1).numpy(),
                 "Si": Si.squeeze(dim=1).numpy(),
                 "Deff": deff.detach().squeeze(dim=1).numpy(),
                 "Dsph": dsph.detach().squeeze(dim=1).numpy()}

        df = pd.DataFrame(data=d)
        df = df.dropna()
        df.to_csv("Fits/"+plotname+"fits_every50th.csv")

        plot_Gc_functionaldependence(deff,dsph,m0[:,0,:],nucleation,plotname)

    return df
def fit_alpha(df,name,loadfile=False):

    pysr_model_alpha = pysr.PySRRegressor(
        niterations=500,  # < Increase me for better results
        model_selection="best",
        binary_operators=["+", "*", "^", "/", "-"],
        unary_operators=[
            "log",
            "exp",
            "inv(x) = 1/x",
            "square",
            "cube",
            # "abs",
            "tanh",
            # "sinh",
            # "cosh"
            # ^ Custom operator (julia syntax)
        ],
        constraints={'^': (-1, 1), 'tanh': 0},
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        verbosity=0,
        output_directory = "PySRfits",
        run_id = name,
        output_torch_format=True,
        # ^ Custom loss function (julia syntax)
    )

    X = df.to_numpy()[:,0:2]
    Yfit = df.to_numpy()[:,2]
    Yexact = df.to_numpy()[:,3]
    if loadfile==False:
        pysr_model_alpha.fit(X, Yfit,
                             variable_names = ["si","T"])
    else:
        PySRfilename = os.path.join("PySRfits",name)
        print("Loading "+PySRfilename)
        pysr_model_alpha=pysr.PySRRegressor.from_file(run_directory=PySRfilename)

    Ypysr = pysr_model_alpha.predict(X)

    print(pysr_model_alpha)
    print(pysr_model_alpha.sympy())

    Si = X[:,0]
    Temp = X[:, 1]
    alpha_nelson = Yexact
    alpha_NN = Yfit
    alpha_SR = Ypysr
    equation = pysr_model_alpha.latex()

    plot_alpha_fits(Temp,Si,alpha_nelson,alpha_NN,alpha_SR,equation,name)

    return pysr_model_alpha

def fit_gc(df,name,loadfile=False):
    pysr_model_ggc = pysr.PySRRegressor(
        niterations=500,  # < Increase me for better results
        model_selection="best",
        binary_operators=["+", "*", "^", "/", "-"],
        # binary_operators=["+", "*","/","-"],
        unary_operators=[
            # "log",
            # "exp",
            "inv(x) = 1/x",
            "square",
            "cube",
            # "abs",
            # "tanh",
            # "sinh",
            # "cosh"
            # ^ Custom operator (julia syntax)
        ],
        constraints={'^': (-1, 1), 'tanh': 0, 'log': 0, 'exp': 0, 'cube': 0, 'square': 0, },
        #dimensional_constraint_penalty=1000,
        #dimensionless_constants_only = False,
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        verbosity=0,
        output_directory="PySRfits",
        run_id=name,
        output_torch_format=True,
        #denoise=True
        # ^ Custom loss function (julia syntax)
    )

    #print(df.head())
    X = np.zeros((len(df), 4))

    # scale these the same as the NN?
    X[:,0] = df.to_numpy()[:,0] * 1e12  # *1e12 #mass
    X[:,1] = df.to_numpy()[:,1]/273.15  # temperature
    X[:,2] = df.to_numpy()[:,2] - 1.0 # Si
    X[:,3] = df.to_numpy()[:,4] * 1e9 # Gc

    Yfit = df.to_numpy()[:,3] * 1e9  # /GGc.to_numpy()[:,5]#*1e9 #

    if loadfile==False:
        pysr_model_ggc.fit(X, Yfit,
                           X_units = ["ng","","","ug/m/s"],
                           y_units = "ug/m/s",
                           variable_names = ["m","T_scaled","si","Gc"])
    else:
        PySRfilename = os.path.join("PySRfits",name)
        print("Loading "+PySRfilename)
        pysr_model_ggc=pysr.PySRRegressor.from_file(run_directory=PySRfilename)

    print(pysr_model_ggc)
    print(pysr_model_ggc.sympy())
    GcSR = pysr_model_ggc.predict(X)*1e-9
    GcNN = Yfit*1e-9
    mass = X[:,0]*1e-12

    plot_Gc_fits(GcSR,GcNN,mass,name)

    return pysr_model_ggc