import numpy as np

def scrita_h2019(T_celsius):
    # these are from Table 1 in Harrington et al. 2019 (https://doi.org/10.1175/JAS-D-18-0319.1)
    # https://journals.ametsoc.org/view/journals/atsc/78/3/JAS-D-20-0228.1.xml
    # c axis (basal facet)
    T0 = 273.15
    T_kelvin = T_celsius + 273.15
    delT = T_kelvin - T0

    x = max(-100.0, min(delT, -1.0))
    # print(x)

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
def satice(temp):
    # saturation vapor pressure with respect to ice at a given temperature
    #  Input: Temperature (K).
    #  Returns: Saturation vapor pressure (Pa).
    # Murphy/Koop equation for saturation vapor pressure over ice (Pa) vs T(K).
    ci = [9.550426, 5723.265, 3.53068, 0.00728332]
    ln_pice = ci[0] - ci[1]/temp + ci[2]*np.log(temp) - ci[3]*temp
    return np.exp(ln_pice)
def satliq(temp):
    # saturation vapor pressure with respect to liquid water at a given temperature
    #  Input: Temperature (K).
    #  Returns: Saturation vapor pressure (Pa).
    C0, C1, C2, C3, C4, C5, C6, C7, C8 = [610.5851, 44.40316, 1.430341, .2641412E-1, .2995057E-3,.2031998E-5, .6936113E-8, .2564861E-11, -.3704404E-13]
    T = temp - 273.16
    return C0+T*(C1+T*(C2+T*(C3+T*(C4+T*(C5+T*(C6+T*(C7+T*C8)))))))
