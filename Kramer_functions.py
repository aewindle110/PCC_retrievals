#Functions to use with spectral pigment derivative algorithm (Kramer et al., 2022)

import numpy as np
from scipy.optimize import fmin


def RInw(lambda_nm, Tc, S):
    
    # % refractive index of air is from Ciddor (1996,Applied Optics)
    n_air = 1.0 + (5792105.0 / (238.0185 - 1 / (lambda_nm / 1e3) ** 2) + 167917.0 / (57.362 - 1 / (lambda_nm / 1e3) ** 2)) / 1e8

    # refractive index of seawater is from Quan and Fry (1994, Applied Optics)
    n0 = 1.31405
    n1 = 1.779e-4
    n2 = -1.05e-6
    n3 = 1.6e-8
    n4 = -2.02e-6
    n5 = 15.868
    n6 = 0.01155
    n7 = -0.00423
    n8 = -4382
    n9 = 1.1455e6
    
    nsw = n0 + (n1 + n2 * Tc + n3 * Tc ** 2) * S + n4 * Tc ** 2 + (n5 + n6 * S + n7 * Tc) / lambda_nm + n8 / lambda_nm ** 2 + n9 / lambda_nm ** 3
    nsw *= n_air
    dnswds = (n1 + n2 * Tc + n3 * Tc ** 2 + n6 / lambda_nm) * n_air

    return nsw, dnswds


def BetaT(Tc, S):
    # pure water secant bulk Millero (1980, Deep-sea Research)
    kw = 19652.21 + 148.4206*Tc - 2.327105*Tc**2 + 1.360477e-2*Tc**3 - 5.155288e-5*Tc**4
    Btw_cal = 1.0 / kw

    # isothermal compressibility from Kell sound measurement in pure water
    # Btw = (50.88630 + 0.717582*Tc + 0.7819867e-3*Tc**2 + 31.62214e-6*Tc**3 - 0.1323594e-6*Tc**4 + 0.634575e-9*Tc**5) / (1 + 21.65928e-3*Tc) * 1e-6

    # seawater secant bulk
    a0 = 54.6746 - 0.603459*Tc + 1.09987e-2*Tc**2 - 6.167e-5*Tc**3
    b0 = 7.944e-2 + 1.6483e-2*Tc - 5.3009e-4*Tc**2

    Ks = kw + a0*S + b0*S**1.5

    # calculate seawater isothermal compressibility from the secant bulk
    IsoComp = 1.0 / Ks * 1e-5  # unit is pa
    return IsoComp

def rhou_sw(Tc, S):
    # density of water and seawater, unit is Kg/m^3, from UNESCO,38,1981
    a0 = 8.24493e-1
    a1 = -4.0899e-3
    a2 = 7.6438e-5
    a3 = -8.2467e-7
    a4 = 5.3875e-9
    a5 = -5.72466e-3
    a6 = 1.0227e-4
    a7 = -1.6546e-6
    a8 = 4.8314e-4
    b0 = 999.842594
    b1 = 6.793952e-2
    b2 = -9.09529e-3
    b3 = 1.001685e-4
    b4 = -1.120083e-6
    b5 = 6.536332e-9
     
    # density for pure water 
    density_w = b0 + b1*Tc + b2*Tc**2 + b3*Tc**3 + b4*Tc**4 + b5*Tc**5
    # density for pure seawater
    density_sw = density_w + ((a0 + a1*Tc + a2*Tc**2 + a3*Tc**3 + a4*Tc**4)*S + (a5 + a6*Tc + a7*Tc**2)*S**1.5 + a8*S**2)
    
    return density_sw

def dlnasw_ds(Tc, S):
    # water activity data of seawater is from Millero and Leung (1976,American
    # Journal of Science,276,1035-1077). Table 19 was reproduced using
    # Eqs.(14,22,23,88,107) then were fitted to polynomial equation.
    # dlnawds is the partial derivative of the natural logarithm of water activity
    # with respect to salinity
    
    dlnawds = (-5.58651e-4 + 2.40452e-7*Tc - 3.12165e-9*Tc**2 + 2.40808e-11*Tc**3) + \
              1.5 * (1.79613e-5 - 9.9422e-8*Tc + 2.08919e-9*Tc**2 - 1.39872e-11*Tc**3) * S**0.5 + \
              2 * (-2.31065e-6 - 1.37674e-9*Tc - 1.93316e-11*Tc**2) * S
    
    return dlnawds

def PMH(n_wat):
    n_wat2 = n_wat ** 2
    n_density_derivative = (n_wat2 - 1) * (1 + 2/3 * (n_wat2 + 2) * (n_wat/3 - 1/(3*n_wat)) ** 2)
    return n_density_derivative

# Xiaodong Zhang, Lianbo Hu, and Ming-Xia He (2009), Scatteirng by pure
# seawater: Effect of salinity, Optics Express, Vol. 17, No. 7, 5698-5710 

# lambda (nm): wavelength
# Tc: temperauter in degree Celsius, must be a scalar
# S: salinity, must be scalar
# delta: depolarization ratio, if not provided, default = 0.039 will be used.
# betasw: volume scattering at angles defined by theta. Its size is [x y],
# where x is the number of angles (x = length(theta)) and y is the number
# of wavelengths in lambda (y = length(lambda))
# beta90sw: volume scattering at 90 degree. Its size is [1 y]
# bw: total scattering coefficient. Its size is [1 y]
# for backscattering coefficients, divide total scattering by 2

# Xiaodong Zhang, March 10, 2009

def betasw_ZHH2009(wave, Tc, theta, S, delta=0.039):
    # Constants
    Na = 6.0221417930e23  # Avogadro's constant
    Kbz = 1.3806503e-23  # Boltzmann constant
    Tk = Tc + 273.15  # Absolute temperature
    M0 = 18e-3  # Molecular weight of water in kg/mol

    if np.isscalar(Tc) and np.isscalar(S):
        Tc = np.array([Tc])
        S = np.array([S])
    else:
        raise ValueError("Both Tc and S must be scalar variables.")

    # Convert input to numpy arrays
    lambda_nm = np.array(wave) #a row variable
    rad = np.deg2rad(theta) #angle in radian as a column variable

    # nsw: absolute refractive index of seawater
    # dnds: partial derivative of seawater refractive index w.r.t. salinity
    nsw, dnds = RInw(lambda_nm, Tc, S)

    # isothermal compressibility is from Lepple & Millero (1971,Deep Sea-Research), pages 10-11
    #The error ~ +/-0.004e-6 bar^-1
    IsoComp = BetaT(Tc, S)
    
    # density of water and seawater,unit is Kg/m^3, from UNESCO,38,1981
    density_sw = rhou_sw(Tc, S)
    
    # water activity data of seawater is from Millero and Leung (1976, American
    # Journal of Science,276,1035-1077). Table 19 was reproduced using Eq.(14,22,23,88,107)
    # then were fitted to polynominal equation.
    # dlnawds is partial derivative of natural logarithm of water activity w.r.t.salinity
    dlnawds = dlnasw_ds(Tc, S)
    
    # density derivative of refractive index from PMH model
    DFRI = PMH(nsw)  # PMH model
    
    # volume scattering at 90 degree due to the density fluctuation
    beta_df = np.pi * np.pi / 2 * ((lambda_nm * 1e-9) ** (-4)) * Kbz * Tk * IsoComp * DFRI ** 2 * (6 + 6 * delta) / (6 - 7 * delta)
    
    # volume scattering at 90 degree due to the concentration fluctuation
    flu_con = S * M0 * dnds ** 2 / density_sw / (-dlnawds) / Na
    beta_cf = 2 * np.pi * np.pi * ((lambda_nm * 1e-9) ** (-4)) * nsw ** 2 * (flu_con) * (6 + 6 * delta) / (6 - 7 * delta)
   
    # total volume scattering at 90 degree
    beta90sw = beta_df + beta_cf
    bsw = 8 * np.pi / 3 * beta90sw * (2 + delta) / (1 + delta)
    betasw = beta90sw[:, np.newaxis] * (1 + ((np.cos(rad)) ** 2) * (1 - delta) / (1 + delta))

    return betasw, beta90sw, bsw

def gsm_cost(IOPs, rrs, aw, bbw, bbpstar, A, B, admstar):
    g = np.array([0.0949, 0.0794])  # orig., constants in eq 2. Gordon et al., 1998
    
    #a = aw + IOPs[0] * A**B  + IOPs[1] * admstar
    a = aw + A*IOPs[0]**B  + IOPs[1] * admstar
    #a = aw + (A*IOPs[0])**B  + IOPs[1] * admstar
    bb = bbw + IOPs[2] * bbpstar
    x = bb / (a + bb)
    rrspred = (g[0] + g[1] * x) * x
    cost = np.sum((rrs - rrspred) ** 2)
    return cost

def gsm_invert(rrs, aw, bbw, bbpstar, A, B, admstar):
    IOPs = np.full((rrs.shape[0], 3), np.nan)
    
    IOPsinit = np.array([0.02, 0.01, 0.0029]) #NOTE THAT THE CHL RETRIEVALS ARE ESPECIALLY SENSITIVE TO INITIAL GUESS!    
    
    rrs_obs = rrs.values
    minimum = fmin(func=gsm_cost, x0= IOPsinit, args=(rrs_obs, aw, bbw, bbpstar, A, B, admstar), xtol=1e-9, ftol=1e-9, maxfun=2000, maxiter=2000, full_output=True, retall=True)

    IOPs = minimum[0]
    return IOPs  