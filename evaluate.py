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
from plotting import plot_Dterm_functionaldependence,plot_dterm_fits, plot_Gc_fits_2panel
from models import massice
from data_utils import getname

from constants import constants,expranges

def evaluate(args,traindata,massratio,model,saveplot=True,learnedfunction=None):
    massratio_real = massratio.float().to(DEVICE)
    Temp = traindata.T.unsqueeze(dim=1).float().to(DEVICE)
    Si = traindata.Si.unsqueeze(dim=1).float().to(DEVICE)
    P = traindata.P.unsqueeze(dim=1).float().to(DEVICE)
    nucleation = traindata.nucleation.unsqueeze(dim=1).float().to(DEVICE)
    r0 = traindata.r0
    explengths = traindata.explength

    nexps = massratio_real.shape[0]

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
    names = [synreal, "NODE", "nelson", "nosurfk"]

    if learnedfunction is not None:
        if args.physics == "strong":  # strong form of the model
            model_symb = massice(physics=args.physics, depmodel="SR",learnedfunction=learnedfunction).float().to(DEVICE)
        elif args.physics == "medium":  # fit dterm
            print("Using medium physics constraint")
            model_symb = massice(physics=args.physics, dtermmodel="SR",learnedfunction=learnedfunction).float().to(DEVICE)
        else:  # weak - fit G transfer coefficient
            print("Using weak physics constraint")
            model_symb = massice(physics=args.physics, gmodel="SR",learnedfunction=learnedfunction).float().to(DEVICE)
        mass_symb = model_symb(m0.to(DEVICE), time.squeeze(dim=0).to(DEVICE), Temp, Si).detach()
        massratios_symb = (mass_symb.cpu().detach() / m0s)[:, :, 0]
        massratios.append(massratios_symb)
        names.append("SR")

    #plotname = "Fits_{}_{}_{:04d}".format(synreal, sw,args.num_iterations)
    plotname = getname(args)
    if args.appendSR is not None:
        plotname = plotname + args.appendSR

    if saveplot==True:
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
            Tempall = torch.broadcast_to(Temp[:, None, :], (nexps, 500, 1)).reshape(nexps * 500, 1)
            Siall = torch.broadcast_to(Si[:, None, :], (nexps, 500, 1)).reshape(nexps * 500, 1)
            massall = (m0s[:, :, :] * massratio[:, :500].unsqueeze(dim=2)).reshape(nexps * 500, 1)
            nuclall = torch.broadcast_to(nucleation[:,None,:],(nexps, 500, 1)).reshape(nexps * 500, 1)


            deff, K = model.dmidt.gmodel.dtermmodel(massall[::50].to(DEVICE), Tempall[::50], Siall[::50])
            dsph, K = model_nosurfk.dmidt.gmodel.dtermmodel(massall[::50].to(DEVICE), Tempall[::50], Siall[::50])
            nucl = nuclall[::50]

            d = {"Mass": massall[::50].detach().squeeze(dim=1).numpy(),
                 "Temp": Tempall[::50].squeeze(dim=1).numpy(),
                 "Si": Siall[::50].squeeze(dim=1).numpy(),
                 "Deff": deff.detach().squeeze(dim=1).numpy(),
                 "Dsph": dsph.detach().squeeze(dim=1).numpy()}
        else:
            deff, K = model.dmidt.gmodel.dtermmodel(m0[:, :].to(DEVICE), Temp, Si)
            dsph, K = model_nosurfk.dmidt.gmodel.dtermmodel(m0[:, :].to(DEVICE), Temp, Si)
            nucl = nucleation


            d = {"Mass": m0.detach().squeeze(dim=1).numpy(),
                 "Temp": Temp.squeeze(dim=1).numpy(),
                 "Si": Si.squeeze(dim=1).numpy(),
                 "Deff": deff.detach().squeeze(dim=1).numpy(),
                 "Dsph": dsph.detach().squeeze(dim=1).numpy()}

        plot_Dterm_functionaldependence(deff, dsph, m0, nucl, plotname)

    else:
        if args.allpoints:  # uses more than just the first points
            npts = args.maxexplength
            it = int(npts/10)
            print(npts,it)
            Tempall = torch.broadcast_to(Temp[:, None, :], (nexps,npts, 1)).reshape(nexps * npts, 1)
            Siall = torch.broadcast_to(Si[:, None, :], (nexps, npts, 1)).reshape(nexps * npts, 1)
            massall = (m0s[:, :npts, :] * massratio[:, :npts].unsqueeze(dim=2)).reshape(nexps *npts, 1)
            massall_NODE = (m0s[:, :npts, :] * massratios_NODE[:, :npts].unsqueeze(dim=2)).reshape(nexps *npts, 1)
            nuclall = torch.broadcast_to(nucleation[:,None,:],(nexps, npts, 1)).reshape(nexps * npts, 1)

            geff  = model.dmidt.gmodel(massall[::it].to(DEVICE), Tempall[::it], Siall[::it])
            gsph = model_nosurfk.dmidt.gmodel(massall[::it].to(DEVICE), Tempall[::it], Siall[::it])
            diff = torch.abs(massall[::it] - massall_NODE[::it])/massall[::it]
            #diff[torch.isinf(diff)] = 1.0
            diff = torch.ones_like(diff)
            nucl = nuclall[::it]
            mpts = massall[::it].detach().squeeze(dim=1).numpy()

            d = {"Mass": massall[::it].detach().squeeze(dim=1).numpy(),
                 "Temp": Tempall[::it].squeeze(dim=1).numpy(),
                 "Si": Siall[::it].squeeze(dim=1).numpy(),
                 "Geff": geff.detach().squeeze(dim=1).numpy(),
                 "Gsph": gsph.detach().squeeze(dim=1).numpy(),
                 "Weights":diff.detach().squeeze(dim=1).numpy(),
                 "nucl":nucl.squeeze(dim=1).numpy()}

        elif args.firstpoints:  # uses the first 10 points
            Tempall = torch.broadcast_to(Temp[:, None, :], (nexps, 10, 1)).reshape(nexps * 10, 1)
            Siall = torch.broadcast_to(Si[:, None, :], (nexps, 10, 1)).reshape(nexps * 10, 1)
            nuclall = torch.broadcast_to(nucleation[:,None,:],(nexps, 10, 1)).reshape(nexps* 10, 1)
            massall = (m0s[:, :10, :] * massratio[:, :10].unsqueeze(dim=2)).reshape(nexps* 10, 1)
            massall_NODE = (m0s[:, :10, :] * massratios_NODE[:, :10].unsqueeze(dim=2)).reshape(nexps * 10, 1)

            geff  = model.dmidt.gmodel(massall.to(DEVICE), Tempall, Siall)
            gsph = model_nosurfk.dmidt.gmodel(massall.to(DEVICE), Tempall, Siall)
            diff = torch.abs(massall- massall_NODE+1e-16)/massall
            diff = torch.ones_like(diff)
            nucl = nuclall
            mpts = massall.detach().squeeze(dim=1).numpy()


            d = {"Mass": massall.detach().squeeze(dim=1).numpy(),
                 "Temp": Tempall.squeeze(dim=1).numpy(),
                 "Si": Siall.squeeze(dim=1).numpy(),
                 "Geff": geff.detach().squeeze(dim=1).numpy(),
                 "Gsph": gsph.detach().squeeze(dim=1).numpy(),
                 "Weights":diff.detach().squeeze(dim=1).numpy(),
                 "nucl":nucl.squeeze(dim=1).numpy()}
        else:
            geff  = model.dmidt.gmodel(m0[:,:].to(DEVICE),Temp, Si)
            gsph = model_nosurfk.dmidt.gmodel(m0[:,:].to(DEVICE),Temp, Si)
            nucl = nucleation
            mpts = m0

            d = {"Mass": m0.detach().squeeze(dim=1).numpy(),
                 "Temp": Temp.squeeze(dim=1).numpy(),
                 "Si": Si.squeeze(dim=1).numpy(),
                 "Geff": geff.detach().squeeze(dim=1).numpy(),
                 "Gsph": gsph.detach().squeeze(dim=1).numpy(),
                 "nucl":nucl.squeeze(dim=1).numpy()}

        plot_Gc_functionaldependence(geff,gsph,mpts,nucl,plotname)

    df = pd.DataFrame(data=d)
    orig = len(df)
    df = df.dropna()
    nonans = len(df)
    print('Dropped '+str(orig-nonans)+ ' rows.')

    df.to_csv("Fits/" + plotname + "_fits.csv")

    return df
def fit_alpha(df,name,args,niterations = 100):

    pysr_model_alpha = pysr.PySRRegressor(
        niterations=niterations,  # < Increase me for better results
        model_selection="best",
        binary_operators=["+", "*", "^", "/", "-"],
        unary_operators=[
            #"log",
            #"exp",
            "inv(x) = 1/x",
            #"square",
            #"cube",
            # "abs",
            "tanh",
            # "sinh",
            # "cosh"
            # ^ Custom operator (julia syntax)
        ],
        constraints={'^': (-1, 1)},
        #constraints={'^': (-1, 1), 'tanh': 6, 'log': 6, 'exp': 6, 'cube': 6, 'square': 6, },
        nested_constraints={'tanh': {'tanh': 0}},
        #nested_constraints={'tanh':{'tanh':0},'log':{'log':0, 'exp':0},'exp':{'log':0,'exp':0}},
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        extra_torch_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        verbosity=1,
        output_directory = "PySRfits",
        run_id = name,
        output_torch_format=True,
        random_state=args.randomseed,
        parsimony = 1e-2,
        maxsize = 25,
        deterministic=True,
        parallelism="serial"
        # ^ Custom loss function (julia syntax)
    )

    outs = df.to_numpy()

    Temp = outs[:, 0]
    Si = outs[:,1]
    Yfit = outs[:, 2]
    Yexact = outs[:, 3]

    nnscaled = False

    if nnscaled == True:
        Tscaled = (Temp - expranges.minTemp) / expranges.Temprange
        Siscaled = (Si - expranges.minSi) / expranges.Sirange
    else:
        Tscaled = Temp
        Siscaled = Si - 1.0

    X = np.column_stack((Tscaled, Siscaled))

    if args.loadSR==False:
        pysr_model_alpha.fit(X, Yfit,
                             variable_names = ["si","T"])
    else:
        PySRfilename = os.path.join("PySRfits",name)
        print("Loading "+PySRfilename)
        pysr_model_alpha=pysr.PySRRegressor.from_file(run_directory=PySRfilename,
                                                      extra_sympy_mappings = {"inv": lambda x: 1 / x},
                                                      extra_torch_mappings = {"inv": lambda x: 1 / x})

    Ypysr = pysr_model_alpha.predict(X)

    print(pysr_model_alpha)
    print(pysr_model_alpha.sympy())


    alpha_nelson = Yexact
    alpha_NN = Yfit
    alpha_SR = Ypysr
    equation = pysr_model_alpha.latex()

    plot_alpha_fits(Temp,Si,alpha_nelson,alpha_NN,alpha_SR,equation,name)

    return pysr_model_alpha
def fit_dterm(df,name,args,niterations = 100):
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
            "sqrt",
            "cbrt",
            # "abs",
            # "tanh",
            # "sinh",
            # "cosh"
            # ^ Custom operator (julia syntax)
        ],
        constraints={'^': (-1, 1), 'sqrt': 0, 'cbrt': 0, 'exp': 0, 'cube': 0, 'square': 0, },
        dimensional_constraint_penalty=1000,
        #dimensionless_constants_only = False,
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        extra_torch_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        verbosity=1,
        output_directory="PySRfits",
        run_id=name,
        output_torch_format=True,
        random_state=args.randomseed,
        deterministic=True,
        parsimony = 1500,
        parallelism="serial"
        #denoise=True
        # ^ Custom loss function (julia syntax)
    )

    outs = df.to_numpy()

    mass = outs[:, 0]
    Temp = outs[:, 1]
    Si = outs[:,2]

    Deff = outs[:, 3]
    Dsph = outs[:, 4]



    mlogscaled = True
    nnscaled = False

    if nnscaled == True:
        Tscaled = (Temp - expranges.minTemp) / expranges.Temprange
        Siscaled = (Si - expranges.minSi) / expranges.Sirange
        if mlogscaled == True:
            mscaled = (np.log(mass) - expranges.minm0_log) / expranges.m0range_log
        else:
            mscaled = (mass - expranges.minm0) / expranges.m0range
    else:
        radius =  (3 * mass / (4 * math.pi * constants.RHOICE)) ** (1 / 3)
        rscaled = radius*1e6
        mscaled = mass*1e12
        Tscaled = Temp
        Siscaled = Si - 1.0
        Deffscaled = Deff*1e6
        Dsphscaled = Dsph*1e6

    X = np.column_stack((rscaled,mscaled,Tscaled, Siscaled,Dsphscaled))

    Yfit = Deffscaled #/Dsph

    if args.loadSR==False:
        pysr_model_dterm.fit(X, Yfit,
                             X_units=["um", "ng", "K", "", "mm^2/s"],
                             #X_units = ["um","ng","K","","m^2/s"],
                             y_units = "mm^2/s",
                             variable_names = ["r_scaled","m_scaled","T_scaled","si_scaled","Dscaled"])

    else:
        PySRfilename = os.path.join("PySRfits",name)
        print("Loading "+PySRfilename)
        pysr_model_dterm=pysr.PySRRegressor.from_file(run_directory=PySRfilename,
                                                      extra_sympy_mappings = {"inv": lambda x: 1 / x},
                                                      extra_torch_mappings = {"inv": lambda x: 1 / x})


    print(pysr_model_dterm)
    print(pysr_model_dterm.sympy())
    DeffSR = pysr_model_dterm.predict(X)*Dsph
    DeffNN = Deff

    plot_dterm_fits(DeffSR,DeffNN,mass,name)

    return pysr_model_dterm

def fit_gc(df,name,args,niterations = 100):
    if args.L1loss:
        loss = "L1DistLoss()"
    else:
        loss = "L2DistLoss()"

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
            #"sqrt",
            #"cbrt",
            #"abs",
            #"tanh",
            # "sinh",
            # "cosh"
            # ^ Custom operator (julia syntax)
        ],
        constraints={'^': (-1, 1), 'sqrt': 0, 'cbrt': 0, 'cube': 0, 'square': 0, },
        #constraints={'^': (-1, 1)},
        # nested_constraints={'sqrt': {'sqrt': 0, 'cbrt':0, 'cube': 0, 'square': 0},
        #                     'cbrt': {'sqrt': 0, 'cbrt':0, 'cube': 0, 'square': 0},
        #                     'cube': {'sqrt': 0, 'cbrt':0, 'cube': 0, 'square': 0},
        #                     'square': {'sqrt': 0, 'cbrt':0, 'cube': 0, 'square': 0},},
        #dimensional_constraint_penalty=0, # 1000 default
        #dimensionless_constants_only = False,
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        extra_torch_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss=loss,
        verbosity=1,
        output_directory="PySRfits",
        run_id=name,
        output_torch_format=True,
        #denoise=True,
        random_state = args.randomseed,
        deterministic = True,
        parallelism = "serial",
        #weight_optimize = 0.001,
        turbo=True,
        #parsimony = 1500 # 1500
        weight_optimize = 0.001,
        #denoise=True
        # ^ Custom loss function (julia syntax)
    )

    outs = df.to_numpy()

    mass = outs[:, 0]
    Temp = outs[:, 1]
    Si = outs[:,2]

    Geff = outs[:, 3]
    Gsph = outs[:, 4]

    mlogscaled = True
    nnscaled = False

    if nnscaled == True:
        Tscaled = (Temp - expranges.minTemp) / expranges.Temprange
        Siscaled = (Si - expranges.minSi) / expranges.Sirange
        if mlogscaled == True:
            mscaled = (np.log(mass) - expranges.minm0_log) / expranges.m0range_log
        else:
            mscaled = (mass - expranges.minm0) / expranges.m0range
    else:
        radius =  (3 * mass / (4 * math.pi * constants.RHOICE)) ** (1 / 3)
        rscaled = radius*1e6
        mscaled = mass*1e12
        Tscaled = Temp #/273.15
        Siscaled = Si - 1.0
        Geffscaled = Geff*1e9
        Gcscaled = Gsph*1e9

    X = np.column_stack((mscaled,Tscaled,Siscaled,Gcscaled))
    #X = np.column_stack((rscaled,Tscaled, Siscaled))

    Yfit = Geffscaled
    #Yfit = Geff/Gsph

    #weights = (Temp - 190) / Temp
    #weights = weights / np.max(weights)
    weights = 1 / (Si * 0.15)
    weights = weights/np.max(weights)
    weights = np.ones_like(Si)
    #if args.allpoints:
    #    weights = 1/(outs[:,5])

    if args.loadSR==False:
        pysr_model_ggc.fit(X, Yfit,
                           X_units = ["ng","K","","kg/nm/s"],
                           y_units = "kg/nm/s",
                           variable_names = ["mscaled","T","si","Gscaled"],
                           weights = weights
                           )

        # pysr_model_ggc.fit(X, Yfit,
        #                    X_units = ["um","K",""],
        #                    y_units = "",
        #                    variable_names = ["rscaled","T","si"])
    else:
        PySRfilename = os.path.join("PySRfits",name)
        print("Loading "+PySRfilename)
        pysr_model_ggc=pysr.PySRRegressor.from_file(run_directory=PySRfilename,
                                                      extra_sympy_mappings = {"inv": lambda x: 1 / x},
                                                      extra_torch_mappings = {"inv": lambda x: 1 / x})

    print(pysr_model_ggc)
    print(pysr_model_ggc.sympy())
    #GcSR = pysr_model_ggc.predict(X)*Gsph

    GcSR = pysr_model_ggc.predict(X)*(1e-9)
    GcNN = Geff

    #print(X[0:10],GcSR[0:10],GcNN[0:10])
    #print("here")
    plot_Gc_fits(GcSR,GcNN,mass,name)
    plot_Gc_fits_2panel(GcSR,GcNN,mass,name,args)
    #print("here")

    return pysr_model_ggc