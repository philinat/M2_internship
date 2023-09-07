"""Fit Non-Stationary GEV models with linear trend in parameters and relaxed shape parameter"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import scipy
import pickle
import dill

from stats.math_functions import LinearExpr
from stats.stats_utils import neg_log_likelihood
from stats.stats_tests import Test_LikelihoodRatio
from stats import VarDistribution, FitDist


### CST ###

sl_init = 10**-2


### FUNC ###

def get_vdistr_norm(model, params, covar):
    """generate a variable distribution"""
    assert isinstance(params, dict), 'parameters must be stored in dict'

    loc = params['loc']
    scale = params['scale']
    c = params['c']

    assert len(loc) == 2 or len(scale) == 2, 'at least one of the parameters must be [slope, intercept] shaped for NS distribution'

    if len(loc) == 2:
        loc_sl = loc[0]
        loc_int = loc[1]
        loc_expr = LinearExpr([loc_sl, loc_int])
    else:
        loc_expr = loc[0]

    if len(scale) == 2:
        scale_sl = scale[0]
        scale_int = scale[1]
        scale_expr = LinearExpr([scale_sl, scale_int])
    else:
        scale_expr = scale[0]

    # normalize covariate
    covar_norm = (covar - np.nanmin(covar)) / (np.nanmax(covar) - np.nanmin(covar))

    inds = np.arange(0, len(covar))

    vdistr = VarDistribution(model, loc=loc_expr, scale=scale_expr, c=c)

    x = pd.DataFrame({"x": covar_norm})
    x.index = covar_norm

    output = vdistr(x)

    output.index = inds

    return output


def compute_nsgev_nllh_m10_freexi_norm(params, model, am_sta):
    """compute the nllh of a non-stationary distribution with time-varying parameters"""
    loc = [params[0], params[1]]
    scale = [params[2]]
    c = params[3]

    params_ = {'loc': loc, 'scale': scale, 'c': c}

    years = am_sta.index
    years_ = years[~years.duplicated()]

    vdistr = get_vdistr_norm(model, params_, years_)

    probs_yrs = []

    for iy, distr in vdistr.items():
        probs = distr.pdf(am_sta.loc[years_[iy]])
        probs = np.asarray([probs])
        probs = probs.flatten()
        probs_yrs.append(probs)

    probs_yrs = np.asarray(probs_yrs)

    ps = np.concatenate(probs_yrs)

    nllhs_ns = neg_log_likelihood(ps)

    return nllhs_ns


def compute_nsgev_nllh_m01_freexi_norm(params, model, am_sta):
    """compute the nllh of a non-stationary distribution with time-varying parameters"""
    loc = [params[0]]
    scale = [params[1], params[2]]
    c = params[3]

    params_ = {'loc': loc, 'scale': scale, 'c': c}

    years = am_sta.index
    years_ = years[~years.duplicated()]

    vdistr = get_vdistr_norm(model, params_, years_)

    probs_yrs = []

    for iy, distr in vdistr.items():
        probs = distr.pdf(am_sta.loc[years_[iy]])
        probs = np.asarray([probs])
        probs = probs.flatten()
        probs_yrs.append(probs)

    probs_yrs = np.asarray(probs_yrs)

    ps = np.concatenate(probs_yrs)

    nllhs_ns = neg_log_likelihood(ps)

    return nllhs_ns


def compute_nsgev_nllh_m11_freexi_norm(params, model, am_sta):
    """compute the nllh of a non-stationary distribution with time-varying parameters"""
    loc = [params[0], params[1]]
    scale = [params[2], params[3]]
    c = params[4]

    params_ = {'loc': loc, 'scale': scale, 'c': c}

    years = am_sta.index
    years_ = years[~years.duplicated()]

    vdistr = get_vdistr_norm(model, params_, years_)

    probs_yrs = []

    for iy, distr in vdistr.items():
        probs = distr.pdf(am_sta.loc[years_[iy]])
        probs = np.asarray([probs])
        probs = probs.flatten()
        probs_yrs.append(probs)

    probs_yrs = np.asarray(probs_yrs)

    ps = np.concatenate(probs_yrs)

    nllhs_ns = neg_log_likelihood(ps)

    return nllhs_ns


def minimize_nsgev_nllh_freexi_norm(params, model, am_sta):
    """minimize the nllh with time-dependent gev parameters"""
    loc = params['loc']
    scale = params['scale']
    c = params['c']

    if len(loc) == 2 and len(scale) == 1:
        nllh_func = compute_nsgev_nllh_m10_freexi_norm

    elif len(loc) == 1 and len(scale) == 2:
        nllh_func = compute_nsgev_nllh_m01_freexi_norm

    elif len(loc) == 2 and len(scale) == 2:
        nllh_func = compute_nsgev_nllh_m11_freexi_norm

    else:
        print('loc and/or scale must be non-stationary')

    paraminit = list(loc) + list(scale) + list(c)

    output = scipy.optimize.minimize(nllh_func, x0=paraminit, args=(model, am_sta), method='Nelder-Mead')

    return output


#def load_nsgev_params_freexi_norm(params=['loc', 'scale']):
    """load fitted NS GEV parameters"""


### MAIN ###

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument("--params", nargs="+", help='params to be tested for non-stationary', type=str, default=['loc', 'scale'])
    parser.add_argument("--model", help='AM distribution model', type=str, default="gev")
    parser.add_argument("--dist_method", help='distribution fit method', type=str, default="mle")

    opts = parser.parse_args()

    params = opts.params
    model = opts.model
    dist_method = opts.dist_method

    print('NS parameters to consider: {0}'.format(params))


    #~ Get data

    #am = REAL AMAX TIME SERIES AS INPUT
    am_test = np.random.rand(100) * 20 + 100    # values without trend (or random trend)
    am = pd.Series(am_test)
    am = am.sort_index()

    years = am.index

    #am = Sample(am)

    #~ Fit S-GEV distribution: get first guess parameter values

    dfit = FitDist(am, model=model, method=dist_method)

    sdist = dfit.distribution
    sparams = sdist.params()
    sloc = sparams['loc']
    sscale = sparams['scale']
    sc = sparams['c']

    nllhs_s = dfit.nllh()

    print('S-GEV nllh: {0}'.format(nllhs_s))

    """sloc = am.mean()
    sscale = am.std()
    sc = -0.01"""

    #~ Fit NS-GEV model

    fg_params = {}

    if 'loc' in params:
        loc_ = [sl_init, sloc]
    else:
        loc_ = [sloc]

    if 'scale' in params:
        scale_ = [sl_init, sscale]
    else:
        scale_ = [sscale]

    c_ = [sc]

    fg_params = {'loc':loc_, 'scale':scale_, 'c': c_}

    print('First guess: {0}'.format(fg_params))


    print('Minimizing nllh...')

    optim = minimize_nsgev_nllh_freexi_norm(params=fg_params, model=model, am_sta=am)


    #~ Treat optim outputs

    res = optim.success
    nllhs_ns = optim.fun
    optim_pars = optim.x

    if ('loc' in params) and ('scale' not in params):

        c_val = [optim_pars[3]]

        if res == True and -1 <= c_val[0] <= 1:
            nllh_val = nllhs_ns
            loc_vals = [optim_pars[0], optim_pars[1]]
            scale_vals = [optim_pars[2]]

        else:
            res = False
            nllh_val = nllhs_s
            loc_vals = [np.nan]
            scale_vals = [np.nan]
            c_val = [np.nan]

    elif ('loc' not in params) and ('scale' in params):

        c_val = [optim_pars[3]]

        if res == True and -1 <= c_val[0] <= 1:
            nllh_val = nllhs_ns
            loc_vals = [optim_pars[0]]
            scale_vals = [optim_pars[1], optim_pars[2]]

        else:
            res = False
            nllh_val = nllhs_s
            loc_vals = [np.nan]
            scale_vals = [np.nan]
            c_val = [np.nan]

    elif ('loc' in params) and ('scale' in params):

        c_val = [optim_pars[4]]

        if res == True and -1 <= c_val[0] <= 1:
            nllh_val = nllhs_ns
            loc_vals = [optim_pars[0], optim_pars[1]]
            scale_vals = [optim_pars[2], optim_pars[3]]

        else:
            res = False
            nllh_val = nllhs_s
            loc_vals = [np.nan]
            scale_vals = [np.nan]
            c_val = [np.nan]

    out_optim = {'succes': res, 'nllh': nllh_val, 'loc': loc_vals, 'scale': scale_vals, 'c': c_val}

    print('output: {0}'.format(out_optim))



    #~ Test significance

    print(Test_LikelihoodRatio(null_loglik=-nllhs_s, alt_loglik=-nllh_val, df_diff=1))

 
    #~ Save

    print('Save')

    params_ = "-".join(params)


    print('Done')


