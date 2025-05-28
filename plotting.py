import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib
import math
import pandas as pd

import glob
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#torch.manual_seed(42)

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from Data.microphysics import satice, satliq


def plotsynthetic(traindata, predmassesReal, predmasses_nelson, massratiosnoise):
    lengths = traindata.explength.clone()
    for i in range(len(traindata)):
        f, (a0, a1) = plt.subplots(2, 1, height_ratios=[3, 1], figsize=(6, 8))

        a0.plot(predmassesReal[i, :lengths[i]], label="Synthetic data", color="k")
        a0.plot(predmasses_nelson[i, :lengths[i]], label="ODE solution", color="r")
        a1.plot(massratiosnoise[i, :lengths[i]], c="k")
        a0.set_ylabel(r"m/$m_{0}$", fontsize=18)
        a1.set_ylabel(r"m/$m_{0}$", fontsize=18)
        a0.set_ylabel(r"m/$m_{0}$", fontsize=18)
        a0.set_title("Simulated Observations", fontsize=18)
        a1.set_title("SSA reconstruction", fontsize=18)
        a0.set_xlabel("Time (s)", fontsize=18)
        a1.set_xlabel("Time (s)", fontsize=18)

        a0.tick_params(axis='both', labelsize=12)
        a1.tick_params(axis='both', labelsize=12)

        a0.legend(frameon=False, fontsize=16)
        plt.tight_layout()
        figname = "Figures/Synthetic/Mratio_Exp{:03d}.png".format(i)
        plt.savefig(figname)
        plt.close()

def plotsynoverview(Temp, Si, alphas_nelson, predmassesReal):
    nexps = len(Temp)

    Trange = np.arange(205, 250, 5)
    satratio = satliq(Trange) / satice(Trange)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    im = axs[0].scatter(Temp, Si, c=alphas_nelson)
    axs[0].set_title(r"$\alpha_{Nelson}$", fontsize=18)
    axs[0].set_ylabel(r"$S_i$", fontsize=18)
    axs[0].set_xlabel("Temperature (K)", fontsize=18)
    axs[0].plot(Trange, satratio, linestyle="dashed", color="k", label="Sat. liq.")
    axs[0].hlines(y=1.0, xmin=Trange[0], xmax=Trange[-1], linestyle="dotted", color="k", label="Sat. ice")
    axs[0].legend(frameon=False)
    axs[0].set_xlim(205, 240)
    # cbar=plt.colorbar(im,label=r"$\alpha$")
    # cb_ax = fig.add_axes([.91,.124,.04,.754])
    plt.colorbar(im).set_label(label=r"$\alpha$", size=15, weight='bold')

    for i in range(0, nexps):
        axs[1].plot(predmassesReal[i, :])
    axs[1].set_title(r"Synthetic data", fontsize=18)
    axs[1].set_ylabel(r"$m/m_{0}$", fontsize=18)
    axs[1].set_xlabel("Time (s)", fontsize=18)
    plt.tight_layout()

    plt.savefig("Figures/Alpha_Nelson_SyntheticDataset.png")
    plt.close()
def compare_mratios_grid(mratios,lengths,names,plotname,ngrids=5,savefig=True):
    # mratios, names should be lists
    colors = ["k", "r", "b", "g", 'y']
    nrows = ngrids
    ncols = ngrids
    nlines = len(mratios)

    nexps = mratios[0].shape[0]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 10))
    index = np.random.choice(nexps, size=nrows * ncols, replace=False)

    axs = axs.ravel()
    for i in range(0, nrows * ncols):
        j = index[i]
        length = int(np.min((499,lengths[j])))
        for k in range(0,nlines):
            axs[i].plot(np.arange(0, length), mratios[k][j, :length], color=colors[k], label=names[k])
            #axs[i].plot(np.arange(0, length), mratio2[j, :length], color="r", linestyle="dashed", label=name2)
        axs[i].set_xlabel("Time (s)", fontsize=12)
        col = "k"
        axs[i].set_ylabel(r"$m/m_{0}$", color=col, fontsize=12)

    #plt.legend(frameon=False)
    fig.legend(names, loc='lower center', ncol=nlines, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=12)

    #plt.tight_layout()
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if savefig==True:
        figname = "Figures/"+plotname+".png"
        plt.savefig(figname)
        plt.close()
    else:
        plt.show()
def plot_alpha_fits(Temp,Si,alpha_nelson,alpha_NN,alpha_SR,equation,name):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(13, 4))

    im = axs[0].scatter(Temp, alpha_nelson, c=Si)
    im = axs[1].scatter(Temp, alpha_NN, c=Si)
    im = axs[2].scatter(Temp, alpha_SR, c=Si)

    axs[0].set_title("Nelson", fontsize=14)
    axs[1].set_title(r"$f_{\alpha}(S_{i},T|\theta_{\alpha})$", fontsize=14)
    axs[2].set_title("Symbolic Regression", fontsize=14)

    eq = r"$"+equation+"$"
    #axs[2].text(0.5, 0.1, eq, horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes,
    #            fontsize=10)
    axs[0].set_ylim(0, 1.0)
    axs[1].set_ylim(0, 1.0)
    axs[2].set_ylim(0, 1.0)

    axs[0].set_xlabel("Temperature (K)", fontsize=14)
    axs[1].set_xlabel("Temperature (K)", fontsize=14)
    axs[2].set_xlabel("Temperature (K)", fontsize=14)
    axs[0].set_ylabel(r"$\alpha$", fontsize=14)

    plt.tight_layout()
    fig.subplots_adjust(right=0.8)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])

    plt.colorbar(im, cax=cbar_ax, label="Si", pad=0.2)

    plt.savefig("Figures/AlphaFuntionalDependence"+name+".png")
    plt.close()
def plot_Gc_fits(GcNN,GcPySR,mass,name):

    fig = plt.figure(figsize=(6, 5))
    plt.title("NODE model vs. symbolic regression")
    plt.scatter(GcNN,GcPySR, c=mass, norm=matplotlib.colors.LogNorm())

    plt.yscale("log")
    plt.xscale("log")
    plt.plot(np.linspace(0, 5) * 1e-9, np.linspace(0, 5) * 1e-9, color="k", linestyle="dashed")
    plt.xlim(5e-11, 1e-8)
    plt.ylim(5e-11, 1e-8)

    cbar = plt.colorbar(label="Mass (kg)")
    cbar.ax.set_ylabel('Mass (kg)', fontsize=14)
    plt.xlabel(r"$G = G_{c}f_{G}(m,S,T|\theta_{G})$", fontsize=18)
    plt.ylabel(r"$G$, Symbolic Regression", fontsize=18)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.tight_layout()

    plt.savefig("Figures/GLearned"+name+".png")
    plt.close()
def plot_Gc_functionaldependence(Geff,Gc,mass,nucleation,name):
    idx0 = np.where(nucleation[:, 0] == 0)  # hetereogenous
    idx1 = np.where(nucleation[:, 0] == 1)  # homogeneous

    plt.scatter(Gc[:, 0].detach()[idx1], Geff[:, 0].detach()[idx1], c=mass[idx1], marker="o",
                label="Homogeneous", norm=matplotlib.colors.LogNorm())

    cbar = plt.colorbar(label="Mass (kg)")
    cbar.ax.set_ylabel('Mass (kg)', fontsize=14)
    plt.scatter(Gc[:, 0].detach()[idx0], Geff[:, 0].detach()[idx0], c=mass[idx0], marker="s",
                norm=matplotlib.colors.LogNorm())
    plt.scatter(Gc[:, 0].detach()[idx0], Geff[:, 0].detach()[idx0], marker="s", label="Heterogeneous",
                facecolor='none', edgecolor='k')
    plt.xlim(3e-11, 7e-9)
    plt.ylim(3e-11, 7e-9)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel(r"$G$", fontsize=18)
    plt.xlabel(r"$G_{c}$", fontsize=18)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.plot(Gc[:, 0].detach(), Gc[:, 0].detach(), linestyle="dashed", color='k')
    plt.legend(frameon=False, loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig("Figures/Gfunction"+name+".png")
    plt.close()