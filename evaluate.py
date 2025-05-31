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
from plotting import plot_Dterm_functionaldependence,plot_dterm_fits
from models import massice
from data_utils import getname

from constants import constants,expranges

def evaluate(args,traindata,massratio,model):
    massratio_real = massratio.float().to(DEVICE)
    Temp = traindata.T.unsqueeze(dim=1).float().to(DEVICE)
    Si = traindata.Si.unsqueeze(dim=1).float().to(DEVICE)
    P = traindata.P.unsqueeze(dim=1).float().to(DEVICE)
    nucleation = traindata.nucleation.unsqueeze(dim=1).float().to(DEVICE)
    r0 = traindata.r0
    explengths = traindata.explength

    m0 = 4 / 3 * math.pi * constants.RHOICE * r0 ** 3
    m0 = m0.unsqueeze(dim=1)
    time = torch.arange(0, args.maxexplength, 1).float()

    model_nelson = massice(physics="strong", depmodel="nelson").float().to(DEVICE)
    model_nosurfk = massice(physics="weak", gmodel="spherical").float().to(DEVICE)

    mass_NODE = model(m0.to(DEVICE), time.squeeze(dim=0).to(DEVICE), Temp, Si).detach()
    mass_nelson = model_nelson(m0.to(DEVICE), time.squeeze(dim=0).to(DEVICE), Temp, Si).detach()
    mass_nosurfk = model_nosurfk(m0.to(DEVICE), time.squeeze(dim=0).to(DEVICE), Temp, Si).detach()

    m0s = m0.unsqueeze(dim=2).expand(mass_NODE.shape)
    massratios_NODE = (mass_NODE.cpu().detach() / m0s)[:, :, 0]
    massratios_nelson = (mass_nelson.cpu().detach() / m0s)[:, :, 0]
    massratios_nosurfk = (mass_nosurfk.cpu().detach() / m0s)[:, :, 0]

    massratios = [massratio_real,massratios_NODE, massratios_nelson, massratios_nosurfk]

    if args.synthetic:
        synreal = "Synthetic"
    else:
        synreal = "Real"
    # if args.strong:
    #     sw = "Strong"
    # else:
    #     sw = "Weak"

    names = [synreal,"NODE", "nelson", "nosurfk"]

    #plotname = "Fits_{}_{}_{:04d}".format(synreal, sw,args.num_iterations)
    plotname = getname(args)
    compare_mratios_grid(massratios, explengths,names, plotname, ngrids=5)

    if args.physics == "strong":
        alphas_pred = model.dmidt.gmodel.dtermmodel.alpha(Si, Temp)
        alphas_nelson = model_nelson.dmidt.gmodel.dtermmodel.alpha(Si, Temp)
        d = {"Temp": Temp.detach().squeeze(dim=1).numpy(),
             "Si": Si.detach().squeeze(dim=1).numpy(),
             "Alpha_pred": alphas_pred.detach().squeeze(dim=1).numpy(),
             "Alpha_Nelson": alphas_nelson.detach().squeeze(dim=1).numpy()}

    elif args.physics == "medium":
        if args.allpoints:  # uses more than just the first few points
            Tempall = torch.broadcast_to(Temp[:, None, :], (290, 500, 1)).reshape(290 * 500, 1)
            Siall = torch.broadcast_to(Si[:, None, :], (290, 500, 1)).reshape(290 * 500, 1)
            massall = (m0s[:, :, :] * massratio[:, :500].unsqueeze(dim=2)).reshape(290 * 500, 1)

            deff, K = model.dmidt.gmodel.dtermmodel(massall[::50].to(DEVICE), Tempall[::50], Siall[::50])
            dsph, K = model_nosurfk.dmidt.gmodel.dtermmodel(massall[::50].to(DEVICE), Tempall[::50], Siall[::50])

            d = {"Mass": massall[::50].detach().squeeze(dim=1).numpy(),
                 "Temp": Tempall[::50].squeeze(dim=1).numpy(),
                 "Si": Siall[::50].squeeze(dim=1).numpy(),
                 "Deff": deff.detach().squeeze(dim=1).numpy(),
                 "Dsph": dsph.detach().squeeze(dim=1).numpy()}
        else:
            deff, K = model.dmidt.gmodel.dtermmodel(m0[:, :].to(DEVICE), Temp, Si)
            dsph, K = model_nosurfk.dmidt.gmodel.dtermmodel(m0[:, :].to(DEVICE), Temp, Si)

            d = {"Mass": m0.detach().squeeze(dim=1).numpy(),
                 "Temp": Temp.squeeze(dim=1).numpy(),
                 "Si": Si.squeeze(dim=1).numpy(),
                 "Deff": deff.detach().squeeze(dim=1).numpy(),
                 "Dsph": dsph.detach().squeeze(dim=1).numpy()}

            plot_Dterm_functionaldependence(deff, dsph, m0, nucleation, plotname)

    else:
        if args.allpoints:  # uses more than just the first few points
            Tempall = torch.broadcast_to(Temp[:, None, :], (290, 500, 1)).reshape(290 * 500, 1)
            Siall = torch.broadcast_to(Si[:, None, :], (290, 500, 1)).reshape(290 * 500, 1)
            massall = (m0s[:, :, :] * massratio[:, :500].unsqueeze(dim=2)).reshape(290 * 500, 1)

            geff  = model.dmidt.gmodel(massall[::50].to(DEVICE), Tempall[::50], Siall[::50])
            gsph = model_nosurfk.dmidt.gmodel(massall[::50].to(DEVICE), Tempall[::50], Siall[::50])

            d = {"Mass": massall[::50].detach().squeeze(dim=1).numpy(),
                 "Temp": Tempall[::50].squeeze(dim=1).numpy(),
                 "Si": Siall[::50].squeeze(dim=1).numpy(),
                 "Geff": geff.detach().squeeze(dim=1).numpy(),
                 "Gsph": gsph.detach().squeeze(dim=1).numpy()}
        else:
            geff  = model.dmidt.gmodel(m0[:,:].to(DEVICE),Temp, Si)
            gsph = model_nosurfk.dmidt.gmodel(m0[:,:].to(DEVICE),Temp, Si)

            d = {"Mass": m0.detach().squeeze(dim=1).numpy(),
                 "Temp": Temp.squeeze(dim=1).numpy(),
                 "Si": Si.squeeze(dim=1).numpy(),
                 "Geff": geff.detach().squeeze(dim=1).numpy(),
                 "Gsph": gsph.detach().squeeze(dim=1).numpy()}

        plot_Gc_functionaldependence(geff,gsph,m0,nucleation,plotname)

    df = pd.DataFrame(data=d)
    orig = len(df)
    df = df.dropna()
    nonans = len(df)
    print('Dropped '+str(orig-nonans)+ ' rows.')

    df.to_csv("Fits/" + plotname + "_fits.csv")

    return df
def fit_alpha(df,name,loadfile=False,niterations = 100):

    pysr_model_alpha = pysr.PySRRegressor(
        niterations=niterations,  # < Increase me for better results
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
        #constraints={'^': (-1, 1), 'tanh': 0},
        constraints={'^': (-1, 1), 'tanh': 0, 'log': 0, 'exp': 0, 'cube': 0, 'square': 0, },
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        extra_torch_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        verbosity=0,
        output_directory = "PySRfits",
        run_id = name,
        output_torch_format=True,
        # ^ Custom loss function (julia syntax)
    )

    outs = df.to_numpy()

    Temp = outs[:, 0]
    Si = outs[:,1]
    Yfit = outs[:, 2]
    Yexact = outs[:, 3]

    Tscaled = (outs[:,0] - expranges.minTemp) / expranges.Temprange
    Siscaled = (outs[:,1] - expranges.minSi) / expranges.Sirange
    X = np.column_stack((Tscaled, Siscaled))

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


    alpha_nelson = Yexact
    alpha_NN = Yfit
    alpha_SR = Ypysr
    equation = pysr_model_alpha.latex()

    plot_alpha_fits(Temp,Si,alpha_nelson,alpha_NN,alpha_SR,equation,name)

    return pysr_model_alpha
def fit_dterm(df,name,loadfile=False,niterations = 100):
    pysr_model_dterm = pysr.PySRRegressor(
        niterations=niterations,  # < Increase me for better results
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
        extra_torch_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        verbosity=0,
        output_directory="PySRfits",
        run_id=name,
        output_torch_format=True,
        #denoise=True
        # ^ Custom loss function (julia syntax)
    )

    outs = df.to_numpy()

    mass = outs[:, 0]
    Temp = outs[:, 1]
    Si = outs[:,2]

    Deff = outs[:, 3]
    Dsph = outs[:, 4]

    mscaled = (np.log(mass) - expranges.minm0) / expranges.m0range
    Tscaled = (Temp - expranges.minTemp) / expranges.Temprange
    Siscaled = (Si - expranges.minSi) / expranges.Sirange
    X = np.column_stack((mscaled,Tscaled, Siscaled))

    Yfit = Deff/Dsph

    if loadfile==False:
        pysr_model_dterm.fit(X, Yfit,
                           X_units = ["ng","",""],
                           y_units = "",
                           variable_names = ["m_scaled","T_scaled","si_scaled"])
    else:
        PySRfilename = os.path.join("PySRfits",name)
        print("Loading "+PySRfilename)
        pysr_model_dterm=pysr.PySRRegressor.from_file(run_directory=PySRfilename)

    print(pysr_model_dterm)
    print(pysr_model_dterm.sympy())
    DeffSR = pysr_model_dterm.predict(X)*Dsph
    DeffNN = Deff

    plot_dterm_fits(DeffSR,DeffNN,mass,name)

    return pysr_model_dterm

def fit_gc(df,name,loadfile=False,niterations = 100):
    pysr_model_ggc = pysr.PySRRegressor(
        niterations=niterations,  # < Increase me for better results
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
        extra_torch_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        verbosity=0,
        output_directory="PySRfits",
        run_id=name,
        output_torch_format=True,
        #denoise=True
        # ^ Custom loss function (julia syntax)
    )

    outs = df.to_numpy()

    mass = outs[:, 0]
    Temp = outs[:, 1]
    Si = outs[:,2]

    Geff = outs[:, 3]
    Gsph = outs[:, 4]

    mscaled = (np.log(mass) - expranges.minm0) / expranges.m0range
    Tscaled = (Temp - expranges.minTemp) / expranges.Temprange
    Siscaled = (Si - expranges.minSi) / expranges.Sirange
    X = np.column_stack((mscaled,Tscaled, Siscaled))

    Yfit = Geff/Gsph

    if loadfile==False:
        pysr_model_ggc.fit(X, Yfit,
                           X_units = ["ng","",""],
                           y_units = "",
                           variable_names = ["m_scaled","T_scaled","si_scaled"])
    else:
        PySRfilename = os.path.join("PySRfits",name)
        print("Loading "+PySRfilename)
        pysr_model_ggc=pysr.PySRRegressor.from_file(run_directory=PySRfilename)

    print(pysr_model_ggc)
    print(pysr_model_ggc.sympy())
    GcSR = pysr_model_ggc.predict(X)*Gsph
    GcNN = Geff

    plot_Gc_fits(GcSR,GcNN,mass,name)

    return pysr_model_ggc