import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import mass_function
import os
from scipy.signal import correlate2d
import pandas as pd
from scipy.stats import truncnorm
from scipy.special import erf
from time import time

path = r'C:\Users\Артем\Documents\GitHub\NN-Clusters/'

df = pd.read_csv(path+r"catalogue_sampler\efeds_data\efeds_exp_map.csv")
df = df[["Texp", "weight"]]
df = df.astype('float32')

sky_map_exp = np.array(df["Texp"], dtype='float32')
sky_map_weight = np.array(df["weight"], dtype='float32')
sky_map_weight /= np.sum(sky_map_weight)

df = pd.read_csv(path+r"catalogue_sampler\efeds_data\lambda_min.csv")
z_lambda_min_array = df["x"].to_numpy()
lambda_min_array = df["y"].to_numpy()

del df

m_max = 10**16
m_min = 10**12

lnM_max = np.log(m_max)
lnM_min = np.log(m_min)

z_min = 0.05
z_max = 1.25

z_piv = 0.35
lnm_piv = np.log(1.4*10**14)

cr_min = 0.02
cr_max = 3
lambda_min = 1
lambda_max = 300
zm_min = 0.1
zm_max = 1.2

def dn_dmdz(lnm, z, cosmo):
    dv_dz = (
    cosmo.angularDiameterDistance(z)**2
    /cosmo.Ez(z)*(1+z)**2
    )*3000
    dn_dmdv = mass_function.massFunction(
        np.exp(lnm), z, mdef='500c', model='bocquet16', q_in='M', 
        q_out='dndlnM', hydro=False
    )
    return dn_dmdv*dv_dz

def lnm_to_l(lnm, z, p_l, cosmo):
    a_l, b_l, delta_l, gamma_l, _ = p_l
    a = np.log(a_l)
    a += gamma_l*np.log((1+z)/(1+z_piv))
    b = b_l + delta_l*np.log((1+z)/(1+z_piv))
    return a + b*(lnm - lnm_piv)

def lnm_to_cr(lnm, z, p_cr, cosmo):
    a_cr, b_cr, delta_cr, gamma_cr, _ = p_cr
    a = np.log(a_cr)
    a += gamma_cr*np.log((1+z)/(1+z_piv))
    a += 2*np.log(cosmo.Ez(z)/cosmo.Ez(z_piv))
    a += -2*np.log(cosmo.luminosityDistance(z)/cosmo.luminosityDistance(z_piv))
    b = b_cr + delta_cr*np.log((1+z)/(1+z_piv))
    return a + b*(lnm - lnm_piv)


def cr_bias(lnm, z, p_bias):
    a_b, b_b, delta_b, gamma_b = p_bias
    a = a_b
    a += gamma_b*np.log(z/z_piv)
    b = b_b + delta_b*np.log(z/z_piv)
    return a + b*(lnm - lnm_piv)


def cr_error(lncr, z, texp):
    return np.exp(
        -0.116 + 0.5789*lncr - 0.393*np.log(texp) - 0.0653*np.log(z)
    )
    
def z_error(z, lambd):
    return np.exp(
        -2.412 + 0.5787*np.log(z) - 0.5*np.log(lambd)
    )
    

def completeness(cr, z, p_comp, cosmo):
    cr50, scr, gz = p_comp
    cr50z = cr50*(
        cosmo.angularDiameterDistance(z)/
        cosmo.angularDiameterDistance(z_piv)
    )**gz
    return 0.5*(1 + erf((np.log(cr) - np.log(cr50z))/scr))

def generate_sample(pars, n_cells=1024):
    params_true = {
        'flat': True, 'H0': pars['h0'], 'Om0': pars['omega_m'], 
        'Ob0': pars['ob'], 'sigma8': pars['sigma_8'], 
        'ns': pars['ns'], 'Tcmb0': 2.7255, 'Neff': 3.046
    }
    cosmo_true = cosmology.setCosmology('true_cosmo', **params_true, persistence = '')
    
    p_cr = [
        pars['a_cr'], pars['b_cr'], pars['delta_cr'],
        pars['gamma_cr'], pars['sigma_cr']
    ]
    p_l = [
        pars['a_l'], pars['b_l'], pars['delta_l'],
        pars['gamma_l'], pars['sigma_l']
    ]
    
    p_bias = [
        pars['a_b'], pars['b_b'], pars['delta_b'], pars['gamma_b']
    ]
    
    p_comp = [
        pars['cr50'], pars['scr'], pars['gz']
    ]

    cov = np.array([
        [pars["sigma_cr"]**2, pars["sigma_cr"]*pars["sigma_l"]*pars['r_cr_l']],
        [pars["sigma_cr"]*pars["sigma_l"]*pars['r_cr_l'], pars["sigma_l"]**2]
    ])

    n_cells = n_cells

    n_points = n_cells + 1

    z_a = np.linspace(z_min, z_max, n_points)
    lnm_a = np.linspace(lnM_min, lnM_max, n_points)
    z_grid, lnm_grid = np.meshgrid(z_a, lnm_a, indexing='ij')
    dz = z_a[1] - z_a[0]
    dlnm = lnm_a[1] - lnm_a[0]

    effective_area = 140*(np.pi/180)**2

    dn_dmdz_grid = np.zeros((n_points, n_points))
    for i in range(n_points):
        dn_dmdz_grid[i] = dn_dmdz(lnm_a, z_a[i], cosmo_true)*effective_area

    kernel = np.ones((2, 2))/4
    dn_grid = correlate2d(dn_dmdz_grid, kernel, mode='valid')*dz*dlnm

    z_grid, lnm_grid = np.meshgrid(
        (z_a[:-1] + z_a[1:])/2, 
        (lnm_a[:-1] + lnm_a[1:])/2, 
        indexing='ij'
    )

    samples = np.random.poisson(lam=dn_grid)

    samples_flat = samples.reshape(-1)
    z_flat = z_grid.reshape(-1)
    lnm_flat = lnm_grid.reshape(-1)
    lncr_flat = lnm_to_cr(lnm_flat, z_flat, p_cr, cosmo_true)
    lncr_flat += cr_bias(lnm_flat, z_flat, p_bias)
    lnl_flat = lnm_to_l(lnm_flat, z_flat, p_l, cosmo_true)

    z_list = []
    lnm_list = []
    lncr_list = []
    lnl_list = []

    for i in range(len(samples_flat)):
        n = int(samples_flat[i])
        if n!=0:
            z_list += [z_flat[i]]*n
            lnm_list += [lnm_flat[i]]*n
            lncr_list += [lncr_flat[i]]*n
            lnl_list += [lnl_flat[i]]*n
        
    cl_sample = np.column_stack((
        z_list, 
        np.column_stack((lncr_list, lnl_list)) + np.random.multivariate_normal(
            np.zeros(2), cov, size=len(lncr_list)
        )
    ))
    
    del z_list, lnm_list, lncr_list, lnl_list, z_flat, lnm_flat, lncr_flat, lnl_flat
    
    cl_sample = cl_sample.astype('float32')

    cl_sample = cl_sample[cl_sample[:, 2]>-3]
    cl_sample = cl_sample[cl_sample[:, 1]>-6.9]

    idxs = np.random.choice(
        len(sky_map_weight), size=len(cl_sample), p=sky_map_weight
    )
    
    cl_sample[:, 1] = np.exp(cl_sample[:, 1]) + np.random.randn(len(cl_sample))*cr_error(
        cl_sample[:, 1], cl_sample[:, 0], sky_map_exp[idxs]
    )
    
    cl_sample = cl_sample[cl_sample[:, 1] > 0]
    
    prob = completeness(
        cl_sample[:, 1], cl_sample[:, 0], p_comp, cosmo_true
    )
    
    prob = np.random.binomial(1, prob)
    
    cl_sample = cl_sample[prob==1]
    
    cl_sample[:, 2] = np.exp(cl_sample[:, 2])
    cl_sample[:, 2] = truncnorm.rvs(
        a=-np.sqrt(cl_sample[:, 2])/1.1, b=np.inf, 
        loc=cl_sample[:, 2], scale=np.sqrt(cl_sample[:, 2])*1.1
    )
    cl_sample[:, 0] = cl_sample[:, 0] + np.random.randn(len(cl_sample))*z_error(cl_sample[:, 0], cl_sample[:, 2])
    
    cl_sample = cl_sample[cl_sample[:, 0]>zm_min]
    cl_sample = cl_sample[cl_sample[:, 0]<zm_max]
    
    cl_sample = cl_sample[cl_sample[:, 2]>np.interp(cl_sample[:, 0], z_lambda_min_array, lambda_min_array)]    

    cl_sample = cl_sample[cl_sample[:, 1]>cr_min]
    cl_sample = cl_sample[cl_sample[:, 1]<cr_max]
    
    if len(cl_sample)==0:
        return np.array([])
    
    return cl_sample
