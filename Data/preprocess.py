# Preprocess the levitation diffusion chamber data sets
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os

from microphysics import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
dir_path = "/Users/karalamb/Columbia/Projects/DepositionalIce/Datasets"

# Cold experiments (4 experiments with high supersaturations, -60 -> - 53 C)

# conditions and calculate Nelson and Baker alpha
Temps = [-63.17,-58.19,-53.99,-60.58]
TempsK = [Temps[i]+273.15 for i in range(0,4)]
Temps_error = [(Temps[i]+273.15)*0.01 for i in range(0,4)]
Si = [1.2111,1.3971,1.5732,1.7404]
Si_error = [0.1*(Si[i]-1.0) for i in range(0,4)]
r0 = [12.16,13.63,9.54,12.09]
r0_m = [r0[i]*1e-6 for i in range(0,4)]
r0_error = [0.27,0.64,0.27,0.37]
r0_errorm = [r0_error[i]*1e-6 for i in range(0,4)]

scrit = [scrita_h2019(Temps[i]) for i in range(0,4)]
alphas = []

for i in range(0,4):
    alpha = ((Si[i]-1)/(scrit[i]))*np.tanh((scrit[i])/(Si[i]-1))
    alphas.append(alpha)

TempsK_cold = np.array(TempsK)
TempsK_error_cold = np.array(Temps_error)
Si_cold = np.array(Si)
Si_error_cold = np.array(Si_error)
r0_m_cold = np.array(r0_m)
r0_m_error_cold = np.array(r0_errorm)
alphas_cold = np.array(alphas)
nucleation_cold = np.ones(4) # homogenous

colnames = ["Time", "Col1", "Col2", "Col3", "Col4",
            "Voltage", "Col6", "Col7", "Col8", "Col9",
            "Col10", "T_bottom", "T_top"]

filenames = ["20210420-130904-crop.dat",
             "20210630-145335-crop.dat",
             "20200825-150358.dat",
             "20200721-150638.dat"]

exps = []
maxlength = 0
for i in range(0, len(filenames)):
    exp = pd.read_csv(os.path.join(dir_path, filenames[i]), names=colnames, sep='\s+')

    # this is to remove some negative values at the beginning of the voltage
    firstnonneg = exp[exp["Voltage"].ge(0)].first_valid_index()
    exp = exp[firstnonneg:]

    times = exp["Time"].values[1:] - exp["Time"].values[1]
    times_int = np.linspace(int(np.min(times)), int(np.max(times)), int(np.max(times)) + 1)

    if len(times_int) > maxlength:
        maxlength = len(times_int)
    exps.append(exp)

massratios_cold = np.zeros((len(exps),maxlength))
times_cold = np.zeros((len(exps),maxlength))
expnames_cold = ["c0","c1","c2","c3"]

explengths_cold = []
for i in range(0, len(exps)):
    # lengthexp = len(exps[i]["Voltage"].values[1:])
    massratios = exps[i]["Voltage"].values[1:] / exps[i]["Voltage"].values[0]
    times = exps[i]["Time"].values[1:] - exps[i]["Time"].values[1]

    times_int = np.linspace(int(np.min(times)), int(np.max(times)), int(np.max(times)) + 1)
    massratios_int = np.interp(times_int, times, massratios)

    lengthexp = len(massratios_int)  # len(exps[i]["Voltage"].values[1:])
    # print(lengthexp)
    # print(massratios_int.shape)

    explengths_cold.append(lengthexp)
    massratios_cold[i, :lengthexp] = massratios_int  # exps[i]["Voltage"].values[1:]/exps[i]["Voltage"].values[0]
    times_cold[i, :lengthexp] = times_int  # exps[i]["Time"].values[1:]-exps[i]["Time"].values[1]

# Data from Pokrifka et al. 2020
subdir_path = "pokrifka-et-al-electrodynamic-levitation-diffusion-homogeneously-nucleated-ice-crystals-2018"

cond_data = "conditions.dat"

colnames =  ["Exp. Num.","Ri(um)","DRi(um)","T(C)","Si(%)","DSi(%)"]
conditions = pd.read_csv(os.path.join(dir_path,subdir_path,cond_data),names = colnames,sep=",")

Temps = conditions["T(C)"].values
TempsK = Temps+273.15
TempsK_error = [(TempsK[i])*0.01 for i in range(0,len(conditions))]
Si = conditions["Si(%)"].values/100.0+1.0
Si_error = conditions["DSi(%)"].values/100.0
r0_m = conditions["Ri(um)"].values*1e-6
r0_m_error = (conditions["DRi(um)"].values)*1e-6

scrit = [scrita_h2019(Temps[i]) for i in range(0,18)]
alphas = []

for i in range(0,18):
    alpha = ((Si[i]-1)/(scrit[i]))*np.tanh((scrit[i])/(Si[i]-1))
    alphas.append(alpha)

TempsK_Pokrifka = np.array(TempsK)
TempsK_error_Pokrifka = np.array(TempsK_error)
Si_Pokrifka = np.array(Si)
Si_error_Pokrifka = np.array(Si_error)
r0_m_Pokrifka = np.array(r0_m)
r0_m_error_Pokrifka = np.array(r0_m_error)
alphas_Pokrifka = np.array(alphas)
nucleation_Pokrifka = np.ones(18) # homogeneous

filenames = ["{:02.0f}".format(i)+"-mass.dat" for i in range(1,19)]
colnames = ["Time(s)","Mass(kg)","Mass ratio","Radius(m)","Volume(m^3)"]

exps = []
maxlength = 0

for i in range(0,len(conditions)):
    filename = filenames[i]

    icedata = pd.read_csv(os.path.join(dir_path,subdir_path,filename),names=colnames,sep="\s+")

    times = icedata["Time(s)"]
    times_int = np.linspace(int(np.min(times)),int(np.max(times)),int(np.max(times))+1)

    if len(times_int)>maxlength:
        maxlength = len(times_int)
    exps.append(icedata)

massratios_Pokrifka = np.zeros((len(exps),maxlength))
times_Pokrifka = np.zeros((len(exps),maxlength))

explengths_Pokrifka = []
for i in range(0, len(exps)):
    massratios = exps[i]["Mass ratio"].values
    times = exps[i]["Time(s)"].values

    times_int = np.linspace(int(np.min(times)), int(np.max(times)), int(np.max(times)) + 1)
    massratios_int = np.interp(times_int, times, massratios)

    lengthexp = len(massratios_int)  # len(exps[i]["Mass ratio"].values)

    explengths_Pokrifka.append(lengthexp)
    massratios_Pokrifka[i, :lengthexp] = massratios_int
    times_Pokrifka[i, :lengthexp] = times_int

# Data from Pokrifka et al. 2023

subdir_path = "lowTdata"
cond_data = "conditions.csv"

colnames =  ["Exp. Num.","Ri(um)","DRi(um)","T(C)","Si(%)","DSi(%)","INP"]
conditions = pd.read_csv(os.path.join(dir_path,subdir_path,cond_data),names = colnames,skiprows=1)

Temps = conditions["T(C)"].values
TempsK = Temps+273.15
TempsK_error = [(TempsK[i])*0.01 for i in range(0,len(conditions))]
Si = conditions["Si(%)"].values/100.0+1.0
Si_error = conditions["DSi(%)"].values/100.0
r0_m = conditions["Ri(um)"].values*1e-6
r0_m_error = (conditions["DRi(um)"].values)*1e-6

ne = len(conditions)
scrit = [scrita_h2019(Temps[i]) for i in range(0,ne)]
alphas = []

for i in range(0,ne):
    alpha = ((Si[i]-1)/(scrit[i]))*np.tanh((scrit[i])/(Si[i]-1))
    alphas.append(alpha)

TempsK_P2023 = np.array(TempsK)
TempsK_error_P2023 = np.array(TempsK_error)
Si_P2023 = np.array(Si)
Si_error_P2023 = np.array(Si_error)
r0_m_P2023 = np.array(r0_m)
r0_m_error_P2023 = np.array(r0_m_error)
alphas_P2023 = np.array(alphas)
# 0 - heterogenous nucleation; 1 - homogenous nucleation
nucleation_P2023 = np.zeros(len(conditions))
nucleation_P2023[np.where(conditions['INP'] == " None")]=1

filenames = ["{:3.0f}".format(i)+"-massratio.csv" for i in range(1,len(conditions)+1)]
colnames = ["Time(s)","Mass ratio","Mass(kg)"]

exps = []
maxlength = 0

for i in range(0,len(conditions)):
    filename = str.strip(filenames[i])

    icedata = pd.read_csv(os.path.join(dir_path,subdir_path,filename),names=colnames,skiprows=1)

    times = icedata["Time(s)"]
    times_int = np.linspace(int(np.min(times)),int(np.max(times)),int(np.max(times))+1)

    if len(times_int)>maxlength:
        maxlength = len(times_int)
    exps.append(icedata)

massratios_P2023 = np.zeros((len(exps),maxlength))
times_P2023 = np.zeros((len(exps),maxlength))

explengths_P2023 = []
for i in range(0, len(exps)):
    massratios = exps[i]["Mass ratio"].values
    times = exps[i]["Time(s)"].values

    times_int = np.linspace(int(np.min(times)), int(np.max(times)), int(np.max(times)) + 1)
    massratios_int = np.interp(times_int, times, massratios)

    lengthexp = len(massratios_int)  # len(exps[i]["Mass ratio"].values)

    explengths_P2023.append(lengthexp)
    massratios_P2023[i, :lengthexp] = massratios_int
    times_P2023[i, :lengthexp] = times_int

# Combine the data sets into numpy arrays
#Environmental Variables
TempsK = np.concatenate((TempsK_cold,TempsK_Pokrifka,TempsK_P2023))
TempsK_error = np.concatenate((TempsK_error_cold,TempsK_error_Pokrifka,TempsK_error_P2023))
Si = np.concatenate((Si_cold,Si_Pokrifka,Si_P2023))
Si_error = np.concatenate((Si_error_cold,Si_error_Pokrifka,Si_error_P2023))
r0_m = np.concatenate((r0_m_cold,r0_m_Pokrifka,r0_m_P2023))
r0_m_error = np.concatenate((r0_m_error_cold,r0_m_error_Pokrifka,r0_m_error_P2023))
alphas = np.concatenate((alphas_cold,alphas_Pokrifka,alphas_P2023))
nucleation = np.concatenate((nucleation_cold,nucleation_Pokrifka,nucleation_P2023))

# Estimate pressure where not given
#meanPress = np.mean(PressPa_Harrison)
meanPress = 97190.470
PressPa_cold = np.ones_like(TempsK_cold)*meanPress
PressPa_Pokrifka = np.ones_like(TempsK_Pokrifka)*meanPress
PressPa_P2023 = np.ones_like(TempsK_P2023)*meanPress

PressPa = np.concatenate((PressPa_cold,PressPa_Pokrifka,PressPa_P2023))

# Experiment lengths
explengths = np.concatenate((np.array(explengths_cold),np.array(explengths_Pokrifka),np.array(explengths_P2023)))

# Combined Time and Mass Ratios - No Harrison
ncold,lencold = massratios_cold.shape
#nHarrison,lenHarrison = massratios_Harrison.shape
nPokrifka,lenPokrifka = massratios_Pokrifka.shape
nP2023,lenP2023 = massratios_P2023.shape

totalexps = ncold+nPokrifka+nP2023
maxlength = np.max(explengths)

massratios = np.zeros((totalexps,maxlength))
times= np.zeros((totalexps,maxlength))

# no harrison
massratios[0:ncold,:lencold]=massratios_cold
times[0:ncold,:lencold]=times_cold

massratios[ncold:(ncold+nPokrifka),:lenPokrifka]=massratios_Pokrifka
times[ncold:(ncold+nPokrifka),:lenPokrifka]=times_Pokrifka

massratios[ncold+nPokrifka:,:lenP2023]=massratios_P2023
times[ncold+nPokrifka:,:lenP2023]=times_P2023

Temprange = np.arange(205,250,5)

satratio = satliq(Temprange)/satice(Temprange)

fig,axs = plt.subplots(nrows = 1,ncols = 2,figsize=(12,5))
axs = axs.ravel()

axs[0].scatter(TempsK[nucleation==0],Si[nucleation==0],marker='o',label="Heterogenous", facecolors='none',edgecolors='blue')
axs[0].scatter(TempsK[nucleation==1],Si[nucleation==1],marker='+',label="Homogeneous",color="red")
axs[0].plot(Temprange,satratio,linestyle="dashed",color="k",label="Sat. liq.")
axs[0].hlines(y=1.0, xmin=Temprange[0],xmax=Temprange[-1],linestyle="dotted",color="k",label="Sat. ice")
axs[0].set_ylabel(r"$S_i$", fontsize=18)
axs[0].set_xlabel("Temperature (K)", fontsize=18)
axs[0].legend(frameon=False)

for i in range(0,totalexps):
    axs[1].plot(times[i,:explengths[i]],massratios[i,:explengths[i]])
axs[1].set_xlabel("Time (s)", fontsize=18)
axs[1].set_ylabel(r"$m/m_{0}$", fontsize=18)
plt.tight_layout()
axs[0].set_xlim(205,240)
plt.savefig("ExperimentOverview.png")

# Create pytorch dataloaders
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class levdiffdata(Dataset):
    def __init__(self):
        self.massratio = torch.tensor(massratios)
        self.time = torch.tensor(times)
        self.explength = torch.tensor(explengths)
        self.T = torch.tensor(TempsK)
        self.dT = torch.tensor(TempsK_error)
        self.P = torch.tensor(PressPa)
        self.Si = torch.tensor(Si)
        self.dSi = torch.tensor(Si_error)
        self.r0 = torch.tensor(r0_m)
        self.dr0 = torch.tensor(r0_m_error)
        self.alpha = torch.tensor(alphas)
        self.nucleation = torch.tensor(nucleation)

    def __len__(self):
        return len(self.T)

    def __getitem__(self, idx):
        explength = self.explength[idx]
        massratio = self.massratio[idx, :explength]
        time = self.time[idx, :explength]
        T = self.T[idx]
        dT = self.dT[idx]
        P = self.P[idx]
        Si = self.Si[idx]
        dSi = self.dSi[idx]
        r0 = self.r0[idx]
        dr0 = self.dr0[idx]
        alpha = self.alpha[idx]
        nucleation = self.nucleation[idx]

        return massratio, time, T, P, Si, r0, alpha, explength, dT, dSi, dr0, nucleation

traindata = levdiffdata()
torch.save(traindata, 'LevDataUncertainty_noHarrison.pth')
traindata_loader = DataLoader(traindata, batch_size=1, shuffle=False)
torch.save(traindata_loader, 'LevData.pth')