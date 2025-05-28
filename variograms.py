import numpy as np
import math
from scipy.optimize import curve_fit

def fit_variogram(bin_centers,sample_vgm,weights,type_vgm,nugget_guess,sill_guess,range_guess):
    """
    Uses scipy's curve_fit function to optimize the parameters of a given variogram model.

    Parameters:
    ------------------------------------------------------------------------------------------------
    - bin_centers: center values of the distance bins
    - sample_vgm: bin-mean values of sample variogram
    - weights: weights for WNLS optimization method 
    - type_vgm: type of variogram to fit (choose between linear, exponential, spherical or gaussian)
    - nugget_guess: initial guess for nugget
    - sill_guess: initial guess for sill
    - range_guess: initial guess for range

    Return:
    ------------------------------------------------------------------------------------------------
    - nugget: optimized value for nugget 
    - sill: optimized value for sill 
    - range: optimized value for range (or pseudo-range for exponential variogram)
    """

    # define input for curve_fit
    xdata = bin_centers
    ydata = sample_vgm
    sigma = 1/weights
    p0 = [nugget_guess,sill_guess,range_guess]
    lower_bounds = [0,0,0]
    upper_bounds = [np.inf,np.inf,np.inf]
    bounds = (lower_bounds,upper_bounds)

    # use curve_fit to estimate parameters with WNLS
    popt, _ = curve_fit(type_vgm,xdata,ydata,p0,sigma,absolute_sigma=True,bounds=bounds)

    # extract parameters from output
    nugget, sill, range = popt

    return nugget, sill, range


def lin_vgm(dist,n,s,r):
    # distance, nugget, sill, range
    gamma = n + (s-n)*(dist/r)
    gamma[np.where(dist<=0)] = 0
    gamma[np.where(dist>=r)] = s
    return gamma

def sph_vgm(dist,n,s,r):
    # distance, nugget, sill, range
    # the quantity s-n is called the partial sill
    gamma = n + (s-n)*(3*dist/(2*r) - dist**3/(2*r**3))
    gamma[np.where(dist<=0)] = 0
    gamma[np.where(dist>=r)] = s
    return gamma

def exp_vgm(dist,n,s,r):
    # dist = distance, n = nugget, s = sill, r = pseudo-range
    # for an exponential model, the pseudo-range is the range at which 95% of the sill is reached.
    gamma = n + (s-n)*(1-np.exp(-3*dist/r))
    gamma[np.where(dist<=0)] = 0
    return gamma

def gauss_vgm(dist,n,s,r):
    # distance, nugget, sill, range
    gamma = n + (s-n)*(1-np.exp(-3*dist**2/r**2))
    gamma[np.where(dist<=0)] = 0
    return gamma

def base_model(type,dist,range):
    if type == "Nug":
        return lin_vgm(dist,1,1,1e-10) # range very small to avoid dividing by zero
    elif type == "Lin":
        return lin_vgm(dist,0,1,range)
    elif type == "Sph":
        return sph_vgm(dist,0,1,range)
    elif type == "Exp":
        return exp_vgm(dist,0,1,range)
    elif type == "Gau":
        return gauss_vgm(dist,0,1,range)
    else:
        raise ValueError(f"{type} is not a valid base model type. Use one out of ['Nug','Lin','Sph','Exp','Gau'].")

def compute_LMC(a,b,c,types,dist,ranges):
    # check if ranges of base models are equal
    if len(np.shape(ranges)) > 1:
        raise ValueError(f"Ranges should be 1-dim (one range per base model), but is {len(np.shape(ranges))}-dim (different ranges for same base model).")

    # check length of arrays
    if not len(a) == len(b) == len(c) == len(types) == len(ranges):
        raise ValueError(f"LMC Coefficients, types and ranges must have same length but have lengths a: {len(a)}, b: {len(b)}, c: {len(c)}, types: {len(types)}, ranges: {len(ranges)}.")

    # compute base models
    bm = np.zeros((len(types),len(dist)))
    for i in range(len(types)):
        bm[i] = base_model(types[i],dist,ranges[i])

    # compute variograms
    gamma_X = np.sum(a[:,np.newaxis]*bm, axis=0)
    gamma_Y = np.sum(b[:,np.newaxis]*bm, axis=0)
    gamma_XY = np.sum(c[:,np.newaxis]*bm, axis=0)

    return gamma_X, gamma_Y, gamma_XY

def check_LMC(a,b,c,ranges):
    """
    Check validity of LMC coefficients in order for G^-1 to exist.

    Parameters:
    ---------------------------------------------------------------
    - a: coefficients (sills) of primary variogram base models 
    - b: coefficients (sills) of secondary variogram base models
    - c: coefficients (sills) of cross-variogram base models
    - ranges: ranges of LMC base models
    """

    # check if ranges of base models are equal
    if len(np.shape(ranges)) > 1:
        raise ValueError(f"Ranges should be 1-dim (one range per base model), but is {len(np.shape(ranges))}-dim (different ranges for same base model).")

    # check length of arrays
    if not len(a) == len(b) == len(c) == len(ranges):
        raise ValueError(f"LMC Coefficients, types and ranges must have same length but have lengths a: {len(a)}, b: {len(b)}, c: {len(c)}, ranges: {len(ranges)}.")

    # conditions (ranges are already checked)
    cond_2 = np.all(a>0)
    cond_3 = np.all(b>0)
    cond_4 = np.all((a*b)>(c*c))

    if cond_2 & cond_3 & cond_4:
        print("Valid LMC") 
    else:
        if not cond_2:
            print("Not all coefficients of a are positive.")
        if not cond_3:
            print("Not all coefficients of b are positive.")
        if not cond_4:
            print("a_j * b_j > (c_j)^2 is not fulfilled for all base models.")
        
        print("Invalid LMC")