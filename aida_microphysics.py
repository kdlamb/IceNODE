import numpy as np

#
# Used by "findCritSats" -

# Constants.
R = 8.31441  # Universal gas constant [J/(mole K)]
Mw = 0.01801528  # Molar mass of water [kg/mole]
Ma = 0.0289647  # Molar mass of dry air [kg/mole]
Rd = R / Ma  # Gas constant of dry air [J/(K kg)]
Rv = R / Mw  # Gas constant of water vapor [J/(K kg)]
surfaceTension = 7.5e-2  # Surface tension of water [N/m]
Cp = 1004.0  # Specific heat capacity of dry air [J/(K kg)]
g = 9.807  # Gravity [m/s^2]
eps = Mw / Ma
To = 273.15
rho_w = 1000.0  # Density of water [kg/m^3]
rho_i = 920.0  # Density of ice [kg/m^3]
vhoff = 2.4  # van't Hoff factor


def findCritSats(tc):
    #
    # Partial clone of routine in FORTRAN parcel model.
    tempcdat = np.zeros(39)
    sigdat_a = np.zeros(39)
    sigdat_c = np.zeros(39)

    if (np.all((sigdat_a == 0.0))):  # Only do the read if the arrays are still zeroed out.
        tmp = np.genfromtxt(fname='libbrescritdev_m15_10um.dat', dtype='float')
        tempcdat, sigdat_a, sigdat_c = tmp[:, 0], tmp[:, 1], tmp[:, 2]

    # Handle limiting cases first.
    if (tc > -1.0):
        return [sigdat_a[0] / 100.0, sigdat_c[0] / 100.0]
    if (tc < -39.0):
        return [sigdat_a[-1] / 100.0, sigdat_c[-1] / 100.0]

    # Search through data arrays only if input T lies between -1.0 and -39.0
    templ = 0.0
    temph = 0.0
    for i in range(len(tempcdat)):
        if (tempcdat[i] <= tc):
            templ = tempcdat[i]
            temph = tempcdat[i - 1]
            il = i
            ih = i - 1
            break

    # Interpolate s_crit values.
    wght = np.abs(templ - tc) / np.abs(templ - temph)
    scrit_a = ((1.0 - wght) * sigdat_a[il] + wght * sigdat_a[ih]) / 100.0
    scrit_c = ((1.0 - wght) * sigdat_c[il] + wght * sigdat_c[ih]) / 100.0

    return [scrit_a, scrit_c]


def getscrit(TK):
    #
    # Python code based on routine from older version of FORTRAN parcel model.
    TC = TK - To
    x = max(-70.0, min(TC, -1.0))
    if (TC < -30.0):
        # Original Magee (high scrit)
        sc = 1.8115 + 0.15585 * x + 0.011569 * x ** 2
        sa = 1.8115 + 0.15585 * x + 0.011569 * x ** 2
        # With Bailey and Hallett
        sc = 3.7955 + 0.10614 * x + 0.0075309 * x ** 2
        sa = sc
    elif (TC >= -30.0 and TC <= -22.0):
        sc = 753.63 + 105.97 * x + 5.5532 * x ** 2 + 0.12809 * x ** 3 + 0.001103 * x ** 4
    elif (TC > -22.0 and TC <= -1.0):
        sc = 1.1217 + 0.038098 * x - 0.083749 * x ** 2 - 0.015734 * x ** 3 - 0.0010108 * x ** 4 \
             - 2.9148e-05 * x ** 5 - 3.1823e-07 * x ** 6
    if (TC >= -30.0 and TC <= -22.0):
        sa = -0.71057 - 0.14775 * x + 0.0042304 * x ** 2
    elif (TC > -22.0 and TC <= -15.0):
        sa = -5.2367 - 1.3184 * x - 0.11066 * x ** 2 - 0.0032303 * x ** 3
    elif (TC > -15.0 and TC <= -10.0):
        sa = 0.34572 - 0.0093029 * x + 0.00030832 * x ** 2  # fit to Nelson & Knight
    elif (TC > -10.0 and TC <= -1.0):
        sa = 0.34572 - 0.0093029 * x + 0.00030832 * x ** 2  # Nelson & Knight
    return [sa / 100.0, sc / 100.0]


def getG(T, Dvs, kts):
    #
    # Specific latent heat of vaporization of water (J/kg) from Equation (15.3), Page 778 of S&P.
    Lv = 2.5e6 * (273.15 / T) ** (0.167 + 3.67e-4 * T)
    #
    # Specific latent heat of fusion of water (J/kg) from Equation (15.4), Page 778 of S&P.
    Lf = 0.3337e6 + 2031.0 * (T - To) - 10.467 * (T - To) ** 2
    #
    # Specific latent heat of sublimation of ice.
    Ls = Lv + Lf
    #
    G = (Rv * T) / (Dvs * e_si(T, False)) + (Ls / (kts * T)) * (Ls / (Rv * T) - 1.0)

    return 1.0 / G


def getDepCoeffs(TK, PPa, ssi, scrit_a, scrit_c, alen, clen,m):
    ei0 = e_si(TK, False)
    Dv = 2.11e-5 * (TK / To) ** 1.94 * (101325.0 / PPa)
    kt = 2.3823e-2 + 7.1177e-5 * (TK - To)
    gtp_stand = getG(TK, Dv, kt)
    vw = np.sqrt(8.0 * R * TK / (np.pi * Mw))
    delta = 6.6e-8 * (101325.0 * TK) / (273.15 * PPa)
    aDelta = alen + delta
    cDelta = clen + delta

    cap = capacitance(alen, clen)
    capDelta = capacitance(aDelta, cDelta)
    C3a = (vw * alen * clen / capDelta) / (4.0 * Dv)
    C3c = (vw * alen ** 2 / capDelta) / (4.0 * Dv)
    #
    # Get alpha for a-axis first, ...
    sisloc_a = Dv / (gtp_stand * Rv * TK / ei0) / (1. / (1. + C3a))
    aresid = 0.1
    slocal_diff_a = ssi / sisloc_a
    slrata_interp = 10. ** (0.935 * np.log10(slocal_diff_a / scrit_a))
    slrata_interp_lim = max(min(slrata_interp, 1.0), 1.0 / (sisloc_a))
    sldiffa_upratio = max(0.0, slrata_interp - 1.) + 1.
    slrata_interp = max(slrata_interp, 1.0 / (sisloc_a)) + aresid * slrata_interp \
                    * (slrata_interp / (1. / sisloc_a)) ** (-2) \
                    * min((slrata_interp / (1. / sisloc_a)) ** 6, 1.0) \
                    - 0.06 * slrata_interp * slrata_interp_lim ** 6 \
                    * sldiffa_upratio ** (-6)

    sloc_a = slocal_diff_a / slrata_interp
    alf_a = (sloc_a / scrit_a) ** m * np.tanh((scrit_a / sloc_a) ** m)
    #
    # then get alpha for the c-axis.
    sisloc_c = Dv / (gtp_stand * Rv * TK / ei0) / (1. / (1. + C3c))
    cresid = 0.1
    slocal_diff_c = ssi / sisloc_c
    slratc_interp = 10. ** (0.935 * np.log10(slocal_diff_c / scrit_c))
    slratc_interp_lim = max(min(slratc_interp, 1.0), 1.0 / (sisloc_c))
    sldiffc_upratio = max(0.0, slratc_interp - 1.) + 1.
    slratc_interp = max(slratc_interp, 1.0 / (sisloc_c)) + cresid * slratc_interp \
                    * (slratc_interp / (1. / sisloc_c)) ** (-2) \
                    * min((slratc_interp / (1. / sisloc_c)) ** 6, 1.0) \
                    - 0.06 * slratc_interp * slratc_interp_lim ** 6 \
                    * sldiffc_upratio ** (-6)
    sloc_c = slocal_diff_c / slratc_interp
    alf_c = (sloc_c / scrit_c) ** m * np.tanh((scrit_c / sloc_c) ** m)
    #

    return [sloc_a, sloc_c, alf_a, alf_c]


#
# Non-vectorized capacitance calculation.
#
#  Inputs:  a- and c-axis lengths (real).
#  Returns:  capacitance (real).
#
def capacitance(a, c):
    phi = c / a
    #
    # Oblate spheroid ...
    if (phi < 1.0):
        return (a ** 2 - c ** 2) ** 0.5 / np.arcsin((1.0 - phi ** 2) ** 0.5)
    # Prolate spheroid ...
    elif (phi > 1.0):
        return (c ** 2 - a ** 2) ** 0.5 / np.log(phi + (phi ** 2 - 1.0) ** 0.5)
    # Sphere ...
    else:
        return a


#
# Vectorized capacitance calculation - 17 April 2023.
#
#  Inputs: Numpy arrays containing a- and c-axis lengths.
#  Returns: Numpy array containing the capacitances.
#
#  Note:  Main program should import Python "warnings" module and run
#		  "warnings.filterwarnings('ignore')" to suppress the warnings
#		  generated when the vectorized square root calculation generates
#		  an imaginary result.
def vecCapacitance(aa, cc):
    C = np.zeros(len(aa))
    phi = cc / aa
    #
    C = np.where(phi < 1.0, np.sqrt(aa ** 2 - cc ** 2) / np.arcsin(np.sqrt(1.0 - phi ** 2)), \
                 np.where(phi > 1.0, np.sqrt(cc ** 2 - aa ** 2) / np.log(phi + np.sqrt(phi ** 2 - 1.0)), aa))
    #
    return C


#
# Saturation vapor pressure over water - based on 'es_new2(T)' from Fortran version
#										 of Lagrangian parcel model code.
#
#  Input: Temperature (K).
#  Returns: Saturation vapor pressure (Pa).
#
def e_sw(TK):
    C0, C1, C2, C3, C4, C5, C6, C7, C8 = [610.5851, 44.40316, 1.430341, .2641412E-1, .2995057E-3, \
                                          .2031998E-5, .6936113E-8, .2564861E-11, -.3704404E-13]
    T = TK - 273.16
    return C0 + T * (C1 + T * (C2 + T * (C3 + T * (C4 + T * (C5 + T * (C6 + T * (C7 + T * C8)))))))


#
# Saturation vapor pressure over ice.
#
#  Inputs:  Temperature(K) & useGG (Boolean, True = use Goff-Gratch equation
#				False = use Murphy/Koop equation).
#  Returns:  Saturation vapor pressure (Pa).
#
def e_si(T, useGG):
    ci = [9.550426, 5723.265, 3.53068, 0.00728332]
    if useGG:
        # Goff Gratch equation for saturation vapor pressure over ice (Pa) vs T(K).
        return 100 * 10 ** (-9.09718 * ((273.16 / T) - 1.0) - 3.56654 * np.log10(273.16 / T) + 0.876793 * (
                1.0 - T / 273.16) + np.log10(6.1071))
    else:
        # Murphy/Koop equation for saturation vapor pressure over ice (Pa) vs T(K).
        ln_pice = ci[0] - ci[1] / T + ci[2] * np.log(T) - ci[3] * T
        return np.exp(ln_pice)


def satrat(TC, si):
    TK = TC + To
    C0, C1, C2, C3, C4, C5, C6, C7, C8 = [54.842763, 6763.22, 4.210, 0.000367, 0.0415, 53.878, 1331.22, 9.44523,
                                          0.014025]
    e_sl = np.exp(C0 - C1 / TK - C2 * np.log(TK) + C3 * TK + np.tanh(C4 * (TK - 218.8)) * (
            C5 - C6 / TK - C7 * np.log(TK) + C8 * TK))
    C0, C1, C2, C3 = [9.550426, 5723.265, 3.53068, 0.00728332]
    e_si = np.exp(C0 - C1 / TK + C2 * np.log(TK) - C3 * TK)
    return si / (e_sl / e_si - 1.)


def val(x0, x1, nx, ix, distType):
    #
    # Returns nx logarithmically or linearly
    #  distributed values between x0 and x1.
    if (nx <= 1):
        print('Error in val ...')
    fact = (float(ix) - 1.0) / (float(nx) - 1.0)
    if (distType == 'lin'):
        return x0 + (x1 - x0) * fact
    else:
        return x0 * (x1 / x0) ** fact
