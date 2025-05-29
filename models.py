import math
import pandas as pd
import torch
import numpy as np
from torch import nn
from torchdiffeq import odeint
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#torch.manual_seed(42)

# values determined from experimental data
minTemp = 207.64999999999998
Temprange = 29.19999999999999
minSi = 1.005
Sirange = 0.7530000000000001
minm0 = -28.301211798926495 #log(min(m0))
maxm0 = -19.465152924804006 #log(50*min(m0))
m0range = maxm0 - minm0

class alphaNN(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(alphaNN, self).__init__()

        self.nh = 20
        self.min = 5.0  # minimum exponent for alpha

        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.nh), nn.ReLU(),
            nn.Linear(self.nh, self.nh), nn.ReLU(), )

        self.lin = nn.Linear(self.nh, output_dim)
        torch.nn.init.normal_(self.lin.weight, mean=0.0, std=0.5)
        torch.nn.init.ones_(self.lin.bias)
        self.minTemp = minTemp
        self.Temprange = Temprange
        self.minSi = minSi
        self.Sirange = Sirange

    def forward(self, S, T):
        # do the normalization here
        Tscaled = (T.float() - self.minTemp) / self.Temprange
        RHscaled = (S.float() - self.minSi) / self.Sirange
        x = torch.concatenate((Tscaled, RHscaled), dim=1)

        alpha = nn.Sigmoid()(self.lin(self.layers(x)))
        logalpha = (alpha - 1.0) * self.min

        return alpha  # logalpha
class linearalpha(nn.Module):
    def __init__(self, c=0.5):
        super(linearalpha, self).__init__()
        self.c = c
        self.minTemp = minTemp
        self.Temprange = Temprange
        self.minSi = minSi
        self.Sirange = Sirange

    def forward(self, S, T):
        alpha = self.c * (S - self.minSi) / self.Sirange + (1.0 - self.c) * (T - self.minTemp) / self.Temprange

        return alpha
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
class learnedalpha(nn.Module):
    def __init__(self, learnedfunction=None):
        super(learnedalpha, self).__init__()
        self.learnedalpha = learnedfunction

    def forward(self, S, T):
        # this is a helper function to take in the pysr output and reshape inputs/outputs for the dmidt model
        x = torch.concatenate((S,T),dim=1)

        alpha = self.learnedalpha(x)

        return alpha.unsqueeze(dim=1)
# the capacitance model, with alpha as an unknown function of temperature and supersaturation
class dmidt(nn.Module):
    def __init__(self, depmodel="nelson", sc="h2016", m=1, c=0.5,learnedfunction=None):
        super(dmidt, self).__init__()

        if depmodel == "nelson":
            self.alpha = alphanelson(sc=sc, m=m)
        elif depmodel == "NN":
            self.alpha = alphaNN()
        elif depmodel == "linear":
            # here c is the slope parameter for the linear function
            self.alpha = linearalpha(c=c)
        elif depmodel == "SR":
            # for loading back in the symbolic regression model
            self.alpha = learnedalpha(learnedfunction=learnedfunction)
        else:  # here c is the constant alpha value
            self.alpha = constantalpha(c=c)

        self.amax = -5.0  # maximum exp. for m0 (i.e. m0 = 1e-12 kg)
        self.amin = 3.0  # minimum exp. for m0 (i.e. m0 = 1e-16 kg)

        # constants
        self.ALPHA_THERM_ICE = 1.0

        self.RH_EQ = 1.0
        self.RHOICE = 910.0  # density of ice (kg/m3)

        self.FV = 1.0
        self.FH = 1.0

        self.R = 8.3144521  # J/(mole/K) - universal gas constant
        self.RV = 461.51  # individual gas constant of water vapor - J/kg/K
        self.RA = 287.05  # individual gas constant of air (Rg/Mgas) - J/kg/K

        self.CP = 1005  #
        self.LS = 2.837e6  # Latent heat of sublimation (J/kg)
        self.mw = 18e-3  # molecular weight of water in kg

        # from Pruppacher and Klett 13-14 and 13-20
        self.lambdaa = 8e-8  # m (8e-6 cm)
        self.deltav = 1.3 * self.lambdaa

        self.deltat = 2.16e-7  # m (2.16e-5cm)

        self.JOULES_IN_A_CAL = 4.187
        self.P = 97190

    def forward(self, x, Temp, Si):

        m0 = x
        RH_ICE = Si
        T = Temp
        P = self.P

        ALPHA_DEP = self.alpha(RH_ICE, T)

        RAD = (3 * m0 / (4 * math.pi * self.RHOICE)) ** (1 / 3)
        RHOA = P / self.RA / T  # density of air

        D1 = self.Diff(T, P)  # diffusivity of water vapour in air
        K1 = self.KA(T)  # thermal conductivity of air

        DSTAR = D1 * self.FV / (
                    RAD / (RAD + self.deltav) + D1 * self.FV / RAD / ALPHA_DEP * torch.sqrt(2 * np.pi / self.RV / T))
        # print(DSTAR.shape,ALPHA_DEP.shape,D1.shape,K1.shape,RAD.shape)
        KSTAR = K1 * self.FH / (
                    RAD / (RAD + self.deltat) + K1 * self.FH / RAD / self.ALPHA_THERM_ICE / self.CP / RHOA * torch.sqrt(
                2 * np.pi / self.RA / T))

        ICEGROWTHRATE = self.R * T / (self.svp_ice(T) * DSTAR * self.mw)
        ICEGROWTHRATE = ICEGROWTHRATE + self.LS / (KSTAR * T) * (self.LS * self.mw / (self.R * T) - 1)

        dmidt = 4e0 * np.pi * RAD * (RH_ICE - self.RH_EQ) / ICEGROWTHRATE

        return dmidt

    def alphas(self, Temp, Si):
        RH_ICE = Si
        T = Temp

        ALPHA_DEP = self.alpha(RH_ICE, T)
        return ALPHA_DEP

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

    def Diff(self, T, P):
        """Diffusion of water vapour in air"""
        # same as used in parcel model
        D1 = 2.11e-5 * (T / 273.15) ** 1.94 * (101325 / P)

        return D1

    def KA(self, T):
        """Thermal conductivity of air"""
        # same as used in parcel model
        ka = (5.69 + 0.017 * (T - 273.15)) * 1e-3 * self.JOULES_IN_A_CAL
        return ka
class learnedgeff(nn.Module):
    def __init__(self, learnedfunction=None):
        super(learnedgeff, self).__init__()
        # this is a helper function to take in the pysr output and reshape/rescale inputs for the dmidtNN model
        self.learnedgeff = learnedfunction

    def forward(self, m, S, T, Gc):
        # rescalings
        # scale these the same as the NN?

        m_scaled = m*1e12
        T_scaled = T/273.15
        si_scaled = S - 1.0
        Gc_scaled = Gc*1e9

        x = torch.concatenate((m_scaled,T_scaled,si_scaled,Gc_scaled),dim=1)

        g_scaled = self.learnedgeff(x)

        geff = g_scaled*1e-9

        return geff.unsqueeze(dim=1)
class dmidtNN(nn.Module):
    def __init__(self, input_dim=3, output_dim=1,geffmodel=None,learnedfunction=None):
        super(dmidtNN, self).__init__()

        self.nh = 20 # was 20
        #self.min = 5.0  # minimum exponent for alpha

        # for loading in the PySR symbolic expression
        self.geffmodel = geffmodel
        self.learnedgeff = learnedgeff(learnedfunction=learnedfunction)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.nh), nn.ReLU(),
            nn.Linear(self.nh, self.nh), nn.ReLU(), )

        self.lin = nn.Linear(self.nh, output_dim)
        torch.nn.init.normal_(self.lin.weight, mean=0.0, std=0.5)
        torch.nn.init.ones_(self.lin.bias)
        self.minTemp = minTemp
        self.Temprange = Temprange
        self.minSi = minSi
        self.Sirange = Sirange


        # self.massrange = 1e-10
        self.minm0 = minm0
        self.maxm0 = maxm0
        self.m0range = maxm0 - minm0

        self.rrange = 1e-6

        self.RH_EQ = 1.0
        self.RHOICE = 910.0  # density of ice (kg/m3)

        self.FV = 1.0
        self.FH = 1.0

        self.R = 8.3144521  # J/(mole/K) - universal gas constant
        self.RV = 461.51  # individual gas constant of water vapor - J/kg/K
        self.RA = 287.05  # individual gas constant of air (Rg/Mgas) - J/kg/K

        self.CP = 1005  #
        self.LS = 2.837e6  # Latent heat of sublimation (J/kg)
        self.mw = 18e-3  # molecular weight of water in kg
        self.JOULES_IN_A_CAL = 4.187
        self.P = 97190

    def forward(self, m, T, S):
        # do the normalization here
        rcurr = (3 * m / (4 * math.pi * self.RHOICE)) ** (1 / 3)

        Deff, Deffsph = self.getdeff(m, T, S)

        if self.geffmodel == "SR":
            Deff = self.learnedgeff(m, T, S, Deffsph)

        dmidt = Deff * (S.float() - 1.0) * rcurr * 4 * math.pi

        return dmidt

    def getdeff(self, m, T, S):
        # do the normalization here
        Tscaled = (T.float() - self.minTemp) / self.Temprange
        RHscaled = (S.float() - self.minSi) / self.Sirange
        mscaled = (torch.log(m).float() - self.minm0) / self.m0range
        # print(mscaled[0:2],self.minm0,self.m0range)
        #rscaled = r.float() / self.rrange
        x = torch.concatenate((mscaled, Tscaled, RHscaled), dim=1)
        # x = torch.concatenate((Tscaled,RHscaled),dim=1)
        P = self.P
        D1 = self.Diff(T, P)  # diffusivity of water vapour in air
        K1 = self.KA(T)  # thermal conductivity of air

        ICEGROWTHRATE = self.R * T / (self.svp_ice(T) * D1 * self.mw)
        ICEGROWTHRATE = ICEGROWTHRATE + self.LS / (K1 * T) * (self.LS * self.mw / (self.R * T) - 1)

        Deffratio = (nn.Sigmoid()(self.lin(self.layers(x)))) * 2.0  # 20.0
        Deffsph = 1.0 / ICEGROWTHRATE
        Deff = Deffratio * Deffsph

        return Deff, Deffsph

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

    def Diff(self, T, P):
        """Diffusion of water vapour in air"""
        # same as used in parcel model
        D1 = 2.11e-5 * (T / 273.15) ** 1.94 * (101325 / P)

        return D1

    def KA(self, T):
        """Thermal conductivity of air"""
        # same as used in parcel model
        ka = (5.69 + 0.017 * (T - 273.15)) * 1e-3 * self.JOULES_IN_A_CAL
        return ka

# for a single ice crystal
class massice(nn.Module):
    def __init__(self, ode_method="rk4", depmodel="nelson", sc="h2016", geffmodel=None,strong=True, m=1, c=0.5,learnedfunction=None):
        super(massice, self).__init__()

        self.Temp = None
        self.Si = None
        self.P = None

        self.method = str(ode_method)
        if strong == True:
            self.dmidt = dmidt(depmodel=depmodel, sc=sc, m=m, c=c,learnedfunction=learnedfunction)
        else:
            self.dmidt = dmidtNN(geffmodel=geffmodel,learnedfunction=learnedfunction)
    def forward(self, z0, ts, Temp, Si):

        self.Temp = Temp
        self.Si = Si


        zs = odeint(self.eval_kernel, z0, ts, method=self.method)
        zs.transpose_(0, 1)

        zs = zs
        return zs

    def eval_kernel(self, t, z):
        z = self.dmidt(z, self.Temp, self.Si)  # this is dmidt

        return z
