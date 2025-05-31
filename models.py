import math
import pandas as pd
import torch
import numpy as np
from erfa import gc2gd
from torch import nn
from torchdiffeq import odeint
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from constants import constants,expranges


# alpha models
class constantalpha(nn.Module):
    def __init__(self, c=1.0):
        super(constantalpha, self).__init__()
        self.c = c

    def forward(self, S, T):
        alpha = self.c

        return torch.nn.Parameter(torch.tensor([alpha]), requires_grad=True)

class alphanelson(nn.Module):
    def __init__(self, sc="h2016", m=1):
        super(alphanelson, self).__init__()

        self.scrit = sc
        self.m = m

    def forward(self, S, T):
        # temperature dependence of critical supersaturation
        sc = self.scrit
        m = self.m

        if sc == "l2009":
            scrit = self.scrita_l2009(T)
        elif sc == "woods":
            scrit = self.scrita_woods(T)
        elif sc == "h2016":
            scrit = self.scrita_h2016(T)
        else:
            sc_h2019, sa_h2019 = self.scrita_h2019(T)
            scrit = sc_h2019

        alphad = torch.pow((S / scrit), m) * torch.tanh(torch.pow((scrit / S), m))
        return alphad

    # Parameterizations for temperature dependence of S_char
    def scrita_l2009(self, T_celsius):
        return 0.00048988 * (torch.pow(torch.abs(T_celsius), 2.5539)) / 100.

    def scrita_woods(self, T_celsius):
        return 0.00033266 * (torch.pow(torch.abs(T_celsius), 3.0999)) / 100.

    def scrita_h2016(self, T_celsius):
        return 0.0096066 * (torch.pow(torch.abs(T_celsius), 1.9171)) / 100.

    def scrita_h2019(self, T_celsius):
        # these are from Table 1 in Harrington et al. 2019 (https://doi.org/10.1175/JAS-D-18-0319.1)
        # https://journals.ametsoc.org/view/journals/atsc/78/3/JAS-D-20-0228.1.xml # c axis (basal facet)
        T0 = 273.15
        T_kelvin = T_celsius + 273.15
        delT = T_kelvin - T0

        x = torch.max(-100.0, torch.min(delT, -1.0))

        if delT < -30:
            sc = 3.7955 + 0.10614 * x + 0.00753 * x ** 2
            sa = sc
        elif (delT >= -30) and (delT <= -22.0):
            sc = 753.63 + 105.97 * x + 5.5532 * x ** 2 + 0.12809 * x ** 3 + 0.001103 * x ** 4
        elif (delT >= -22) and (delT <= -1.0):
            sc = 1.1217 + 0.038098 * x - 0.083749 * x ** 2 - 0.015734 * x ** 3 \
                 - 0.0010108 * x ** 4 - 2.9148e-05 * x ** 5 - 3.1823e-07 * x ** 6
        elif delT > -1.0:
            sc = 1.1217 + 0.038098 * x - 0.083749 * x ** 2 - 0.015734 * x ** 3 \
                 - 0.0010108 * x ** 4 - 2.9148e-05 * x ** 5 - 3.1823e-07 * x ** 6
        if (delT >= -30.0) and (delT <= -22.0):
            sa = -0.71057 - 0.14775 * x + 0.0042304 * x ** 2
        elif (delT >= -22) and (delT <= -15.0):
            sa = -5.2367 - 1.3184 * x - 0.11066 * x ** 2 - 0.0032303 * x ** 3
        elif (delT >= -15.0) and (delT <= -1.0):
            sa = 0.34572 - 0.0093029 * x + 0.00030832 * x ** 2  # fit to Nelson & Night
        elif delT > -1.0:
            sa = 0.34572 - 0.0093029 * x + 0.00030832 * x ** 2  # fit to Nelson & Night
        sc = sc / 100.0
        sa = sa / 100.0

        return sc
class alphaNN(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, nhidden=20, learnedfunction=None):
        super(alphaNN, self).__init__()

        self.learnedalpha = learnedfunction

        self.nh = nhidden
        self.min = 5.0  # minimum exponent for alpha

        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.nh), nn.ReLU(),
            nn.Linear(self.nh, self.nh), nn.ReLU(), )

        self.lin = nn.Linear(self.nh, output_dim)
        torch.nn.init.normal_(self.lin.weight, mean=0.0, std=0.5)
        torch.nn.init.ones_(self.lin.bias)

    def forward(self, S, T):
        # do the normalization here
        Tscaled = (T.float() - expranges.minTemp) / expranges.Temprange
        RHscaled = (S.float() - expranges.minSi) / expranges.Sirange
        x = torch.concatenate((Tscaled, RHscaled), dim=1)

        if self.learnedalpha is not None:
            alpha = self.learnedalpha(x)
            alpha = alpha.unsqueeze(dim=1)
        else:
            alpha = nn.Sigmoid()(self.lin(self.layers(x)))
            # logalpha = (alpha - 1.0) * self.min

        return alpha
class dtermNN(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, nhidden=20, learnedfunction=None):
        super(dtermNN, self).__init__()
        # this is a helper function to take in the pysr output and reshape/rescale inputs for the dmidtNN model
        self.learneddterm = learnedfunction
        self.dterm = dterm()

        self.nh = nhidden

        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.nh), nn.ReLU(),
            nn.Linear(self.nh, self.nh), nn.ReLU(), )

        self.lin = nn.Linear(self.nh, output_dim)
        torch.nn.init.normal_(self.lin.weight, mean=0.0, std=0.5)
        torch.nn.init.ones_(self.lin.bias)

    def forward(self, m, T, S):
        # rescalings
        # scale these the same as the NN?

        Tscaled = (T.float() - expranges.minTemp) / expranges.Temprange
        RHscaled = (S.float() - expranges.minSi) / expranges.Sirange
        mscaled = (torch.log(m).float() - expranges.minm0) / expranges.m0range

        x = torch.concatenate((mscaled, Tscaled, RHscaled), dim=1)

        D, K = self.dterm(m, T, S)
        if self.learneddterm is not None:
            Dtermratio = self.learneddterm(x).unsqueeze(dim=1)
        else:
            Dtermratio = (nn.Sigmoid()(self.lin(self.layers(x)))) * 2.0  # 20.0
            #Dtermratio = (nn.Softplus()(self.lin(self.layers(x))))

        return Dtermratio * D, K
class geffNN(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, nhidden=20, learnedfunction=None):
        super(geffNN, self).__init__()
        # this is a helper function to take in the pysr output and reshape/rescale inputs for the dmidtNN model
        self.learnedgeff = learnedfunction
        self.gsph = geffmodel(dtermmodel="diffusive")

        self.nh = nhidden

        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.nh), nn.ReLU(),
            nn.Linear(self.nh, self.nh), nn.ReLU(), )

        self.lin = nn.Linear(self.nh, output_dim)
        torch.nn.init.normal_(self.lin.weight, mean=0.0, std=0.5)
        torch.nn.init.ones_(self.lin.bias)

    def forward(self, m, T, S):
        # rescalings
        # scale these the same as the NN?

        Tscaled = (T.float() - expranges.minTemp) / expranges.Temprange
        RHscaled = (S.float() - expranges.minSi) / expranges.Sirange
        mscaled = (torch.log(m).float() - expranges.minm0) / expranges.m0range
        #print(torch.min(T),torch.min(S),torch.min(m))
        #print(torch.min(Tscaled),torch.min(RHscaled),torch.min(mscaled))
        x = torch.concatenate((mscaled, Tscaled, RHscaled), dim=1)

        if self.learnedgeff is not None:
            Geffratio = self.learnedgeff(x).unsqueeze(dim=1)
        else:
            Geffratio = (nn.Sigmoid()(self.lin(self.layers(x)))) * 2.0  # 20.0

        return Geffratio*self.gsph(m, T, S)
class dterm(nn.Module):
    def __init__(self, translational=False, depmodel="nelson", sc="h2016", m=1, c=0.5, learnedfunction=None):
        super(dterm, self).__init__()

        # continuum or translational case
        self.translational = translational

        # deposition model
        if depmodel == "nelson":
            self.alpha = alphanelson(sc="h2016", m=1)
        elif depmodel == "NN":
            self.alpha = alphaNN()
        elif depmodel == "SR":
            # for loading back in the symbolic regression model
            self.alpha = alphaNN(learnedfunction=learnedfunction)
        else:  # here c is the constant alpha value
            self.alpha = constantalpha(c=c)

    def forward(self, m, T, S):

        P = constants.P
        ALPHA_DEP = self.alpha(S, T)

        RAD = (3 * m / (4 * math.pi * constants.RHOICE)) ** (1 / 3)
        RHOA = P / constants.RA / T  # density of air

        D1 = self.Diff(T, P)  # diffusivity of water vapour in air
        K1 = self.KA(T)  # thermal conductivity of air

        if self.translational == True:
            D = D1 * constants.FV / (RAD / (RAD + constants.deltav) + D1 * constants.FV / RAD / ALPHA_DEP * torch.sqrt(
                2 * np.pi / constants.RV / T))
            K = K1 * constants.FH / (RAD / (RAD + constants.deltat) + K1 * constants.FH / RAD /
                                     constants.ALPHA_THERM_ICE / constants.CP / RHOA * torch.sqrt(
                        2 * np.pi / constants.RA / T))
        else:
            D = D1
            K = K1

        return D, K
    def Diff(self, T, P):
        """Diffusion of water vapour in air"""
        # same as used in parcel model
        D1 = 2.11e-5 * (T / 273.15) ** 1.94 * (101325 / P)
        return D1
    def KA(self, T):
        """Thermal conductivity of air"""
        # same as used in parcel model
        ka = (5.69 + 0.017 * (T - 273.15)) * 1e-3 * constants.JOULES_IN_A_CAL
        return ka
class geffmodel(nn.Module):
    def __init__(self, dtermmodel="diffusive", depmodel="nelson", sc="h2016", m=1, c=0.5, learnedfunction=None):
        super(geffmodel, self).__init__()

        # medium case
        if dtermmodel == "diffusive":
            self.dtermmodel = dterm()
        elif dtermmodel == "translational":
            self.dtermmodel = dterm(translational=True, depmodel=depmodel, sc=sc, m=m, c=c,
                                    learnedfunction=learnedfunction)
        elif dtermmodel == "NN":
            self.dtermmodel = dtermNN()
        elif dtermmodel == "SR":
            self.dtermmodel = dtermNN(learnedfunction=learnedfunction)

    def forward(self, m, T, S):

        RAD = (3 * m / (4 * math.pi * constants.RHOICE)) ** (1 / 3)
        D, K = self.dtermmodel(m, T, S)

        ICEGROWTHRATE = constants.R * T / (self.svp_ice(T) * D * constants.mw)
        ICEGROWTHRATE = ICEGROWTHRATE + constants.LS / (K * T) * (constants.LS * constants.mw / (constants.R * T) - 1)

        gterm = 1 / ICEGROWTHRATE

        return gterm

    def svp_ice(self, T):
        """saturation vapour pressure over ice """
        # the version used in the ACPIM code
        # svp = 100*6.1115e0*torch.exp((23.036e0 - (T-273.15)/333.7e0)*(T-273.15)/(279.82e0 + (T-273.15)))

        # Murphy and Koop, 2005 - same as the parcel model
        # saturation vapor pressure over ice, after Murphy and Koop (2005), in Pa
        # T>110 K
        # temp - K, vp_ice - hPa
        a0 = 9.550426
        a1 = 5723.265
        a2 = 3.53068
        a3 = 0.00728332

        svp = torch.exp(a0 - a1 / T + a2 * torch.log(T) - a3 * T)

        return svp
# the capacitance model, with alpha as an unknown function of temperature and supersaturation
class dmidt(nn.Module):
    def __init__(self, gmodel="spherical", dtermmodel="diffusive", depmodel="nelson", sc="h2016", m=1, c=0.5,
                 learnedfunction=None):
        super(dmidt, self).__init__()

        if gmodel == "spherical":
            self.gmodel = geffmodel()
        elif gmodel == "translational":
            self.gmodel = geffmodel(dtermmodel=dtermmodel, depmodel=depmodel, sc=sc, m=m, c=c,
                                    learnedfunction=learnedfunction)
        elif gmodel == "NN":
            self.gmodel = geffNN()
        elif gmodel == "SR":  # substitute in the learned symbolic regression expression
            self.gmodel = geffNN(learnedfunction=learnedfunction)

    def forward(self, m, T, S):

        RAD = (3 * m / (4 * math.pi * constants.RHOICE)) ** (1 / 3)
        geff = self.gmodel(m, T, S)

        dmidt = 4e0 * np.pi * RAD * (S - 1.0) * geff

        return dmidt
    # for a single ice crystal
class massice(nn.Module):
    def __init__(self, ode_method="rk4", physics="strong", gmodel="spherical", dtermmodel="diffusive",
                 depmodel="nelson",
                 sc="h2016", m=1, c=0.5, learnedfunction=None):
        super(massice, self).__init__()

        self.Temp = None
        self.Si = None

        self.method = str(ode_method)
        if physics == "strong":
            self.dmidt = dmidt(gmodel="translational",
                               dtermmodel="translational",
                               depmodel=depmodel, sc=sc, m=m, c=c,
                               learnedfunction=learnedfunction)
        elif physics == "medium":
            self.dmidt = dmidt(gmodel="translational",
                               dtermmodel=dtermmodel,
                               learnedfunction=learnedfunction)
        else:  # weak constraint
            self.dmidt = dmidt(gmodel=gmodel,
                               learnedfunction=learnedfunction)

    def forward(self, z0, ts, Temp, Si):

        self.Temp = Temp  # +self.dT
        self.Si = Si

        zs = odeint(self.eval_kernel, z0, ts, method=self.method)
        zs.transpose_(0, 1)

        zs = zs
        return zs

    def eval_kernel(self, t, z):
        z = self.dmidt(z, self.Temp, self.Si)

        return z