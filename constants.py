class expranges:
    # values determined from experimental data
    minTemp = 200.0 #207.64999999999998
    Temprange = 40.0 #29.19999999999999
    minSi = 1.000 #1.005
    Sirange = 1.000 #0.7530000000000001
    minm0 = -28.301211798926495 #log(min(m0))
    maxm0 = -19.465152924804006 #log(50*min(m0))
    m0range = maxm0 - minm0

class constants:
    # constants
    ALPHA_THERM_ICE = 1.0

    RH_EQ = 1.0
    RHOICE = 910.0  # density of ice (kg/m3)

    FV = 1.0
    FH = 1.0

    R = 8.3144521  # J/(mole/K) - universal gas constant
    RV = 461.51  # individual gas constant of water vapor - J/kg/K
    RA = 287.05  # individual gas constant of air (Rg/Mgas) - J/kg/K

    CP = 1005  #
    LS = 2.837e6  # Latent heat of sublimation (J/kg)
    mw = 18e-3  # molecular weight of water in kg

    # from Pruppacher and Klett 13-14 and 13-20
    lambdaa = 8e-8  # m (8e-6 cm)
    deltav = 1.3 * lambdaa

    deltat = 2.16e-7  # m (2.16e-5cm)
    JOULES_IN_A_CAL = 4.187

    # constant pressure value
    P = 97190.0