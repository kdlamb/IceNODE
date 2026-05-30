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
from constants import constants, labConfig
from aida_microphysics import getDepCoeffs, vecCapacitance, e_si
from data_utils import getname


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
def plot_alpha_fits(Temp, Si, alpha_nelson, alpha_NN, alpha_SR, equation, name):
    # 2 rows x 3 cols: top = alphas, bottom = residuals
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(13, 7),
                            sharex='col',
                            gridspec_kw={'height_ratios': [3, 2]})

    # --- Top row: raw alpha vs T ---
    sc0 = axs[0,0].scatter(Temp, alpha_nelson, c=Si)
    sc1 = axs[0,1].scatter(Temp, alpha_NN,      c=Si)
    sc2 = axs[0,2].scatter(Temp, alpha_SR,      c=Si)

    axs[0,0].set_title("Nelson", fontsize=14)
    axs[0,1].set_title(r"$f_{\alpha}(S_{i},T|\theta_{\alpha})$", fontsize=14)
    axs[0,2].set_title("Symbolic Regression", fontsize=14)

    for j in range(3):
        axs[0,j].set_ylim(0, 1.0)
        axs[0,j].set_xlabel("Temperature (K)", fontsize=14)
    axs[0,0].set_ylabel(r"$\alpha$", fontsize=14)

    # --- Bottom row: residuals (Nelson - model) vs T ---
    diff_nelson = alpha_nelson - alpha_nelson        # zeros (sanity check)
    diff_nn     = alpha_nelson - alpha_NN
    diff_sr     = alpha_nelson - alpha_SR

    axs[1,0].scatter(Temp, diff_nelson, c=Si)
    axs[1,1].scatter(Temp, diff_nn,     c=Si)
    axs[1,2].scatter(Temp, diff_sr,     c=Si)

    # Symmetric y-lims for residuals
    max_abs = np.nanmax(np.abs(np.concatenate([
        np.atleast_1d(diff_nn), np.atleast_1d(diff_sr)
    ])))
    max_abs = 1e-6 if not np.isfinite(max_abs) or max_abs == 0 else max_abs  # avoid zero
    for j in range(3):
        axs[1,j].axhline(0.0, color='k', linestyle=':', linewidth=1)
        axs[1,j].set_ylim(-max_abs*2, max_abs*2)
        axs[1,j].set_xlabel("Temperature (K)", fontsize=14)
    axs[1,0].set_ylabel(r"$\Delta \alpha$ (Nelson − model)", fontsize=14)

    # Optional: label the residual titles
    # axs[1,0].set_title("Residual: Nelson − Nelson", fontsize=12)
    # axs[1,1].set_title("Residual: Nelson − NN", fontsize=12)
    # axs[1,2].set_title("Residual: Nelson − SymReg", fontsize=12)

    # Equation (if you want to show it)
    # eq = r"$" + equation + "$"
    # axs[0,2].text(0.5, 0.1, eq, ha='center', va='center',
    #               transform=axs[0,2].transAxes, fontsize=10)

    # Shared colorbar (Si), spanning both rows
    plt.tight_layout()
    fig.subplots_adjust(right=0.83, hspace=0.35)
    cbar_ax = fig.add_axes([0.86, 0.15, 0.02, 0.7])
    # use the last scatter handle for the colorbar scale (Si)
    plt.colorbar(sc2, cax=cbar_ax, label="Si", pad=0.2)

    # Panel labels (a–f)
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    for lab, ax in zip(labels, axs.ravel()):
        ax.text(-0.12, 1.02, lab, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='bottom', ha='left')

    #plt.savefig(f"Figures/AlphaFuntionalDependenceRes{name}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"Figures/AlphaFuntionalDependenceRes{name}.pdf", dpi=400, format='pdf',bbox_inches='tight')
    plt.close()
# def plot_alpha_fits(Temp,Si,alpha_nelson,alpha_NN,alpha_SR,equation,name):
#     fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(13, 4))
#
#     im = axs[0].scatter(Temp, alpha_nelson, c=Si)
#     im = axs[1].scatter(Temp, alpha_NN, c=Si)
#     im = axs[2].scatter(Temp, alpha_SR, c=Si)
#
#     axs[0].set_title("Nelson", fontsize=14)
#     axs[1].set_title(r"$f_{\alpha}(S_{i},T|\theta_{\alpha})$", fontsize=14)
#     axs[2].set_title("Symbolic Regression", fontsize=14)
#
#     eq = r"$"+equation+"$"
#     #axs[2].text(0.5, 0.1, eq, horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes,
#     #            fontsize=10)
#     axs[0].set_ylim(0, 1.0)
#     axs[1].set_ylim(0, 1.0)
#     axs[2].set_ylim(0, 1.0)
#
#     axs[0].set_xlabel("Temperature (K)", fontsize=14)
#     axs[1].set_xlabel("Temperature (K)", fontsize=14)
#     axs[2].set_xlabel("Temperature (K)", fontsize=14)
#     axs[0].set_ylabel(r"$\alpha$", fontsize=14)
#
#     plt.tight_layout()
#     fig.subplots_adjust(right=0.8)
#
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
#
#     plt.colorbar(im, cax=cbar_ax, label="Si", pad=0.2)
#
#     plt.savefig("Figures/AlphaFuntionalDependence"+name+".png")
#     plt.close()
def plot_Gc_fits(GcNN,GcPySR,mass,name):

    fig = plt.figure(figsize=(6, 5))
    plt.title("NODE model vs. symbolic regression")
    plt.scatter(GcNN,GcPySR, c=mass, norm=matplotlib.colors.LogNorm())

    plt.yscale("log")
    plt.xscale("log")
    plt.plot(np.linspace(0, 5) * 1e-9, np.linspace(0, 5) * 1e-9, color="k", linestyle="dashed")
    plt.xlim(3e-11, 7e-9)
    plt.ylim(3e-11, 7e-9)

    cbar = plt.colorbar(label="Mass (kg)")
    cbar.ax.set_ylabel('Mass (kg)', fontsize=14)
    plt.xlabel(r"$G = G_{c}f_{G}(m,S,T|\theta_{G})$", fontsize=18)
    plt.ylabel(r"$G$, Symbolic Regression", fontsize=18)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.tight_layout()

    #plt.savefig("Figures/GLearned"+name+".png")
    plt.savefig(f"Figures/GLearned{name}.pdf", dpi=400, format='pdf', bbox_inches='tight')
    plt.close()
def plot_Gc_fits_2panel(GcNN,GcPySR,mass,name,args):
    df_name = getname(args)+".csv"
    try:
        df = pd.read_csv(os.path.join("Dataframes",df_name))
        Geff = df["Geff"]
        Gc = df["Gsph"]
        mass = df["Mass"]
        nucleation = df["nucl"]

        print(Geff.shape, Gc.shape, mass.shape, nucleation.shape)
    except:
        print("No data frame named "+os.path.join("Dataframes",df_name))
        Geff = np.zeros(2,10)
        Gc = np.zeros(2,10)
        mass = np.zeros(2,10)
        nucleation = np.zeros(2,10)

    idx0 = np.where(nucleation == 0)
    idx1 = np.where(nucleation == 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5),constrained_layout=True)
    norm = matplotlib.colors.LogNorm()

    sc = axs[0].scatter(Gc.iloc[idx1],Geff.iloc[idx1],c=mass.iloc[idx1],marker="o",
        label="Homogeneous",norm=norm)
    axs[0].scatter(Gc.iloc[idx0],Geff.iloc[idx0],c=mass.iloc[idx0],marker="s",norm=norm)
    axs[0].scatter(Gc.iloc[idx0],Geff.iloc[idx0],marker="s",label="Heterogeneous",
        facecolor="none",edgecolor="k")
    axs[0].plot(Gc,Gc,linestyle="dashed",color="k")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlim(3e-11, 7e-9)
    axs[0].set_ylim(3e-11, 7e-9)
    axs[0].set_xlabel(r"$G_c$", fontsize=18)
    axs[0].set_ylabel(r"$G$", fontsize=18)
    axs[0].legend(frameon=False, loc="upper left", fontsize=12)
    axs[0].text(-0.12, 1.05, "A",transform=axs[0].transAxes,fontsize=18,fontweight="bold")

    axs[1].scatter(GcNN,GcPySR,c=mass,norm=norm)
    axs[1].plot(np.linspace(0, 5) * 1e-9,np.linspace(0, 5) * 1e-9,color="k",linestyle="dashed")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlim(3e-11, 7e-9)
    axs[1].set_ylim(3e-11, 7e-9)
    axs[1].set_xlabel(r"$G = G_c f_G(m,S,T|\theta_G)$",fontsize=18)
    axs[1].set_ylabel(r"$G$, Symbolic Regression",fontsize=18)
    axs[1].text(-0.12, 1.05, "B",transform=axs[1].transAxes,fontsize=18,fontweight="bold")

    for ax in axs:
        ax.set_box_aspect(1)

    cbar = fig.colorbar(sc,ax=axs,pad=0.02)
    cbar.ax.set_ylabel("Mass (kg)",fontsize=14)

    #plt.tight_layout()

    plt.savefig(f"Figures/Gcomparison{name}.pdf",dpi=400,format="pdf",bbox_inches="tight")
    plt.close()

def plot_dterm_fits(DcNN,DcPySR,mass,name):

    fig = plt.figure(figsize=(6, 5))
    plt.title("NODE model vs. symbolic regression")
    plt.scatter(DcNN,DcPySR, c=mass, norm=matplotlib.colors.LogNorm())

    plt.yscale("log")
    plt.xscale("log")
    plt.plot(np.linspace(0, 3) * 1e-5, np.linspace(0, 3) * 1e-5, color="k", linestyle="dashed")
    plt.xlim(1e-6, 3.0e-5)
    plt.ylim(1e-6, 3.0e-5)

    cbar = plt.colorbar(label="Mass (kg)")
    cbar.ax.set_ylabel('Mass (kg)', fontsize=14)
    plt.xlabel(r"$D = D_{c}f_{D}(m,S,T|\theta_{D})$", fontsize=18)
    plt.ylabel(r"$d$, Symbolic Regression", fontsize=18)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.tight_layout()

    plt.savefig("Figures/DLearned"+name+".png")
    plt.close()

def plot_Gc_functionaldependence(Geff,Gc,mass,nucleation,name):
    print(Geff.shape,Gc.shape,mass.shape,nucleation.shape)

    idx0 = np.where(nucleation[:, 0] == 0)  # hetereogenous
    idx1 = np.where(nucleation[:, 0] == 1)  # homogeneous
    fig = plt.figure(figsize=(6, 5))
    plt.title("Weakly constrained NODE model")
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
    #plt.savefig("Figures/Gfunction"+name+".png")
    plt.savefig(f"Figures/Gfunction{name}.pdf", dpi=400, format='pdf', bbox_inches='tight')
    plt.close()
def plot_Dterm_functionaldependence(Deff,Dc,mass,nucleation,name):
    idx0 = np.where(nucleation[:, 0] == 0)  # hetereogenous
    idx1 = np.where(nucleation[:, 0] == 1)  # homogeneous

    plt.scatter(Dc[:, 0].detach()[idx1], Deff[:, 0].detach()[idx1], c=mass[idx1], marker="o",
                label="Homogeneous", norm=matplotlib.colors.LogNorm())

    cbar = plt.colorbar(label="Mass (kg)")
    cbar.ax.set_ylabel('Mass (kg)', fontsize=14)
    plt.scatter(Dc[:, 0].detach()[idx0], Deff[:, 0].detach()[idx0], c=mass[idx0], marker="s",
                norm=matplotlib.colors.LogNorm())
    plt.scatter(Dc[:, 0].detach()[idx0], Deff[:, 0].detach()[idx0], marker="s", label="Heterogeneous",
                facecolor='none', edgecolor='k')
    #plt.xlim(3e-11, 7e-9)
    #plt.ylim(3e-11, 7e-9)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel(r"$D$", fontsize=18)
    plt.xlabel(r"$D_{c}$", fontsize=18)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.plot(Dc[:, 0].detach(), Dc[:, 0].detach(), linestyle="dashed", color='k')
    plt.legend(frameon=False, loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig("Figures/Dfunction"+name+".png")
    plt.close()


def saveTimeSeries(tsFileName, parceloutputs, inputs, observations, qvc, vecC):
    #
    # Parcel model outputs.
    eps = constants.eps
    qv_out = parceloutputs[0, :]
    si_out = parceloutputs[1, :]
    r_out = parceloutputs[2, :]
    a_out = parceloutputs[3, :]
    c_out = parceloutputs[4, :]
    integratedIce = parceloutputs[5, :]
    IWC = parceloutputs[6, :]
    Tout = parceloutputs[7, :]
    Pout = parceloutputs[8, :]
    qi_out = parceloutputs[9, :]
    q_tot = qi_out + qv_out
    #
    # AIDA chamber measurements.
    Tin = inputs['tg_k']
    Pin = inputs['p_pa']
    Nin = inputs['cn_ice']
    Sin = observations['Si']
    IWC_in = observations['IWC']
    #
    #
    with open(tsFileName, 'w') as f:
        f.write('t Tin Pin Nin Si_in IWC_in Tout Pout Nout Si_out IWC_out qv_out qi_out qtot_out r_out a_out c_out\n')
        fmt = '%6.1f' + 16*' %13.6e' + '\n'
        for i in range(len(Tin)):
            f.write(fmt % (float(i), Tin[i], Pin[i], Nin[i], Sin[i], IWC_in[i], Tout[i], Pout[i], integratedIce[i], si_out[i], \
                    IWC[i]/eps, qv_out[i], qi_out[i], q_tot[i], r_out[i], a_out[i], c_out[i]))


def makeAIDAExpoverviewplot(savename, parceloutputs, inputs, observations, nBinsUsed, savefig=True):
    Rd = constants.RA
    Rv = constants.RV
    eps = constants.eps

    plt.figure(figsize=(30.0, 15.0))
    ax1 = plt.subplot(5, 2, 1)
    ax2 = plt.subplot(5, 2, 2)
    ax3 = plt.subplot(5, 2, 3)
    ax4 = plt.subplot(5, 2, 4)
    ax5 = plt.subplot(5, 2, 5)
    ax6 = plt.subplot(5, 2, 6)
    ax7 = plt.subplot(5, 2, 7)
    ax8 = plt.subplot(5, 2, 8)
    #ax9 = plt.subplot(5, 2, 9)
    #ax10 = plt.subplot(5, 2, 10)

    qv_out = parceloutputs[0, :]
    si_out = parceloutputs[1, :]
    r_out = parceloutputs[2, :]
    a_out = parceloutputs[3, :]
    c_out = parceloutputs[4, :]
    integratedIce = parceloutputs[5, :]
    IWC = parceloutputs[6, :]
    Tout = parceloutputs[7, :]
    Pout = parceloutputs[8, :]
    qi_out = parceloutputs[9, :]

    rhoair = Pout / (Rd * Tout)
    ax1.plot(Tout, linestyle='dashed', color='red')
    ax1.plot(inputs['tg_k'], color='red')
    ax2.plot(Pout, linestyle='dashed', color='blue')
    ax2.plot(inputs['p_pa'], color='blue')
    ax3.plot(si_out[0:-2], linestyle='dashed', color='black', label='Model')
    ax3.plot(observations['Si'], color='black', label='Obs.')
    ax3.axhline(y=1.0, linestyle='dashed', color='r')

    ax4.plot(IWC[0:-2] / eps, color='blue', linestyle='dashed', label="Model")
    ax4.plot(observations['IWC'], color='blue', label="Obs.")
    ax5.plot(integratedIce[0:-2] / 1e6, color='green', linestyle='dashed')
    ax5.plot(inputs['cn_ice'], color='green')
    ax6.plot(qv_out[0:-2], color='green', linestyle='dashed', label='Model')
    qvcalc = si_out * e_si(Tout, False) / Pout * eps
    qvcalc = observations['Si'] * e_si(Tout, False) / Pout * eps
    ax6.plot(qvcalc[0:-2], color='black', label='Obs.')
    ax7.plot(r_out[0:-2], color='black', label='r')
    vc = vecCapacitance(a_out, c_out)
    ax7.plot(a_out[0:-2], color='red', label='a')
    ax7.plot(c_out[0:-2], color='blue', label='c')
    ax7.plot(vc[0:-2], color='green', label="Cap.")

    #ax7.axhline(y=AIDAparcel.inRadius * 1e6, linestyle="dashed", color="k", label='Init.')
    ax7.legend()

    ax8.plot(qi_out[0:-2], color='blue', linestyle='dashed', label='qi')
    q_tot = qi_out + qv_out
    ax8.plot(q_tot[0:-2], color='magenta', label='qt')
    ax8.axhline(y=0.0, linestyle='dashed', color='red')
    ax8.legend()
    # ax9.plot(AIDAparcel.alpha_a[0:nBinsUsed], label='alpha a')
    # ax9.plot(AIDAparcel.alpha_c[0:nBinsUsed], label='alpha c')
    # ax9.legend()
    # ax10.plot(AIDAparcel.slocal_a[0:nBinsUsed])
    # ax10.plot(AIDAparcel.slocal_c[0:nBinsUsed])
    # ax8.plot(observations['IWC']*eps, color='blue')
    ax1.set_ylabel(r'T (K)')
    ax2.set_ylabel(r'P (Pa)')
    ax3.set_ylabel(r'S$_{i}$')
    ax4.set_ylabel(r'IWC (ppmv)')
    ax5.set_ylabel(r'$N_{Ice}$ (#/cm$^{3}$)')
    ax6.set_ylabel(r'$q_{v}$ (kg/kg)')
    ax7.set_ylabel(r'$r_{i}$  ($\mu$m)')
    ax8.set_ylabel(r'$q_{i}$ (kg/kg)')
    #ax9.set_ylabel(r'$\alpha$')
    #ax10.set_ylabel(r'$s_{local}$')
    fig = plt.gcf()
    fig.suptitle(savename, fontsize=14)

    #
    # Show AIDA chamber measurement uncertainties as shaded regions.
    si_obsMax = (1.0 + labConfig.uncSice) * observations['Si']
    si_obsMin = (1.0 - labConfig.uncSice) * observations['Si']
    xValues = np.linspace(0.0, len(si_obsMax) - 1, len(si_obsMax))
    ax3.fill_between(xValues, si_obsMin, si_obsMax, color='darkgreen', alpha=0.2,
                     label=r'$\pm$ ' + str(labConfig.uncSice * 100.0) + ' %')
    ax3.legend()
    #
    T_obsMax = inputs['tg_k'] + labConfig.uncT
    T_obsMin = inputs['tg_k'] - labConfig.uncT
    ax1.fill_between(xValues, T_obsMin, T_obsMax, color='darkgreen', alpha=0.2,
                     label=r'$\pm$ ' + str(labConfig.uncT) + ' K')
    ax1.legend()
    #
    IWC_obsMax = (1.0 + labConfig.uncIWC) * observations['IWC']
    IWC_obsMin = (1.0 - labConfig.uncIWC) * observations['IWC']
    ax4.fill_between(xValues, IWC_obsMin, IWC_obsMax, color='darkgreen', alpha=0.2,
                     label=r'$\pm$ ' + str(labConfig.uncIWC * 100.0) + ' %')
    ax4.legend()
    #
    Nice_obsMax = (1.0 + labConfig.uncNice) * inputs['cn_ice']
    Nice_obsMin = (1.0 - labConfig.uncNice) * inputs['cn_ice']
    ax5.fill_between(xValues, Nice_obsMin, Nice_obsMax, color='darkgreen', alpha=0.2,
                     label=r'$\pm$ ' + str(labConfig.uncNice * 100.0) + ' %')
    ax5.legend()
    #
    qvcalc_max = si_obsMax * e_si(Tout + labConfig.uncT, False) / (Pout - labConfig.uncP) * eps
    qvcalc_min = si_obsMin * e_si(Tout - labConfig.uncT, False) / (Pout + labConfig.uncP) * eps
    ax6.fill_between(xValues, qvcalc_min, qvcalc_max, color='darkgreen', alpha=0.2)
    ax6.legend()

    #
    # Optionally save the timeseries ...
    if labConfig.saveTimeSeries:
        saveTimeSeries('timeSeries.dat', parceloutputs, inputs, observations, qvcalc, vc)
    #
    #  and save the overview plot as a pdf.
    if savefig:
        #print("Saving plots in ", savename)
        plt.savefig(savename)
    else:
        plt.show()
    plt.close()
def plot_complexity_loss(df,name,pysrmodule,savefig=True):
    regressor = pysrmodule
    complexity = regressor.equations['complexity']
    losses = regressor.equations['loss']
    score = regressor.equations['score']
    idx = regressor.equations_.index

    c0 = df.columns[0]
    c1 = df.columns[2]

    plt.axvline(x=df[c0][0], color="r", linestyle="dashed")
    plt.axhline(y=df[c1][0], label="NN", color="r", linestyle="dashed")
    plt.axvline(x=df[c0][1], color="g", linestyle="dashed")
    plt.axhline(y=df[c1][1], label="Nelson", color="g", linestyle="dashed")
    plt.axvline(x=df[c0][2], color="b", linestyle="dashed")
    plt.axhline(y=df[c1][2], label="No Surf. K", color="b", linestyle="dashed")
    plt.scatter(df[c0][3::4], df[c1][3::4], c=complexity[1:], label="SR")
    plt.ylabel("MSE Loss (1500 s)", fontsize=18)
    plt.xlabel("MSE Loss (500 s)", fontsize=18)
    plt.yscale("log")
    plt.xscale("log")
    for j, (cp, ls) in enumerate(zip(df[c0][3::4], df[c1][3::4])):
        plt.text(cp, ls, str(idx[j]), fontsize=10)
    cbar = plt.colorbar()
    cbar.set_label('Complexity', fontsize=18)
    plt.legend()
    plt.tight_layout()

    if savefig:
        plt.savefig("Figures/Complexity_Loss"+name+".png")
    else:
        plt.show()
    plt.close()

    c0 = df.columns[3]
    c1 = df.columns[7]

    plt.axvline(x=df[c0][0], color="r", linestyle="dashed")
    plt.axhline(y=df[c1][0], label="NN", color="r", linestyle="dashed")
    plt.axvline(x=df[c0][1], color="g", linestyle="dashed")
    plt.axhline(y=df[c1][1], label="Nelson", color="g", linestyle="dashed")
    plt.axvline(x=df[c0][2], color="b", linestyle="dashed")
    plt.axhline(y=df[c1][2], label="No Surf. K", color="b", linestyle="dashed")
    plt.scatter(df[c0][3::4], df[c1][3::4], c=complexity[1:], label="SR")
    plt.ylabel("# Best (1500 s)", fontsize=18)
    plt.xlabel("# Best (500 s)", fontsize=18)
    #plt.yscale("log")
    #plt.xscale("log")
    for j, (cp, ls) in enumerate(zip(df[c0][3::4], df[c1][3::4])):
        plt.text(cp, ls, str(idx[j]), fontsize=10)
    cbar = plt.colorbar()
    cbar.set_label('Complexity', fontsize=18)
    plt.legend()
    plt.tight_layout()

    if savefig:
        plt.savefig("Figures/Complexity_Loss_ByExp_"+name+".png")
    else:
        plt.show()
    plt.close()


def plot_sr_scores(pysrmodule, name,savefig=True):
    regressor = pysrmodule
    offx = 0.5
    offy = 0.002
    neq = len(regressor.equations_)

    complexity = regressor.equations['complexity']
    losses = regressor.equations['loss']
    score = regressor.equations['score']
    idx = regressor.equations_.index

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    ax[0].scatter(complexity, losses, c=idx)
    #ax[0].scatter(complexity[i], losses[i], c='r', marker='x')
    for j, (cp, ls) in enumerate(zip(complexity, losses)):
        ax[0].text(cp, ls + offy, str(idx[j]), fontsize=10)
    ax[0].set_xlabel("Complexity", fontsize=18)
    ax[0].set_ylabel("MSE Loss", fontsize=18)

    ax1 = ax[1].scatter(score, losses, c=idx)

    #ax[1].scatter(score[i], losses[i], c='r', marker='x')
    for j, (sc, ls) in enumerate(zip(score, losses)):
        ax[1].text(sc, ls + offy, str(idx[j]), fontsize=10)
    ax[1].set_xlabel("Scores", fontsize=18)

    if savefig:
        plt.savefig("Figures/Complexity_Scores" + name + ".png")
    else:
        plt.show()
    plt.close()