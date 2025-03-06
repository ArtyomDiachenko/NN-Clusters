import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import mass_function
import os
from scipy.signal import correlate2d
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import GPy
from scipy.stats import truncnorm
import joblib
from sklearn.ensemble import RandomForestRegressor
from math import ceil
from time import time

path = r'C:\Users\Артем\Documents\GitHub\NN-Clusters/'

os.environ["OMP_NUM_THREADS"] = "1"

k = GPy.kern.RBF(
    input_dim=4,
    variance=1,
    lengthscale=np.ones(4),
    ARD=True,
    name="rbf"
)

likelihood = GPy.likelihoods.Bernoulli()

m = GPy.core.SVGP(
    np.array([[1, 1, 1, 1]]),
    np.array([[1]]),
    np.random.rand(30, 4),
    kernel=k, 
    likelihood=likelihood,
    batchsize=2**10,
)

m[:] = np.load(path+"catalogue_sampler/selection_function_fit/model_parameters.npy")

scaler_min, scaler_scale = np.load(path+"catalogue_sampler/selection_function_fit/scaler_parameters.npy")
scaler = MinMaxScaler()
scaler.min_ = scaler_min
scaler.scale_ = scaler_scale

def selection_function(x, chunksize=2**15):
    x = scaler.transform(x)
    p = np.zeros(len(x))
    n_chunks = ceil(len(x)/chunksize)
    for i in range(n_chunks-1):
        p[i*chunksize:(i+1)*chunksize] = m.predict(x[i*chunksize:(i+1)*chunksize])[0].reshape(-1)
    
    if (i+1)*chunksize<len(x):
        p[(i+1)*chunksize:] = m.predict(x[(i+1)*chunksize:])[0].reshape(-1)
    
    return p

area_array = np.load(path+"catalogue_sampler/data/effective_area/effective_area.npy")
zlim_array = np.load(path+"catalogue_sampler/data/effective_area/z_array.npy")

df = pd.read_csv(path+"catalogue_sampler/data/skymap/skykoords.csv")
df = df[["Texp", "nh", "weight"]]
df = df.astype('float32')

sky_map_exp = np.array(df["Texp"], dtype='float32')
sky_map_nh = np.array(df["nh"], dtype='float32')
sky_map_weight = np.array(df["weight"], dtype='float32')

del df

m_max = 10**16
m_min = 10**12

lnM_max = np.log(m_max)
lnM_min = np.log(m_min)

z_min = 0.05
z_max = 0.85

z_piv = 0.35
lnm_piv = np.log(1.4*10**14)

cr_min = 0.02
cr_max = 20
lambda_min = 3
lambda_max = 300
zm_min = 0.1
zm_max = 0.8

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


def cr_error(lncr, z, texp):
    return np.exp(
        -0.0756 + 0.385*lncr - 0.429*np.log(texp) - 0.162*np.log(z)
    )
    

z_err_model = joblib.load(path+"catalogue_sampler/data/error_model/redshift_error_model.joblib")

def generate_sample(pars, n_cells=1024):
    t1 = time()
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

    effective_area = np.interp(z_a, zlim_array, area_array)*(np.pi/180)**2

    dn_dmdz_grid = np.zeros((n_points, n_points))
    for i in range(n_points):
        dn_dmdz_grid[i] = dn_dmdz(lnm_a, z_a[i], cosmo_true)*effective_area[i]

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
    

    prob = selection_function(np.column_stack((
        cl_sample[:, 1],
        np.log(sky_map_exp[idxs]),
        np.log(sky_map_nh[idxs]),
        cl_sample[:, 0]
    )))
    

    prob = np.random.binomial(1, prob)
    cl_sample = np.column_stack((
        cl_sample,
        sky_map_exp[idxs]
    ))

    cl_sample = cl_sample[prob==1]
    cl_sample[:, 2] = np.exp(cl_sample[:, 2]) 
    cl_sample[:, 2] = truncnorm.rvs(a=0, b=np.inf, loc=cl_sample[:, 2], scale=cl_sample[:, 2])
    cl_sample = cl_sample[cl_sample[:, 2]>lambda_min]
    cl_sample = cl_sample[cl_sample[:, 2]<lambda_max]

    cl_sample[:, 1] = (
        np.exp(cl_sample[:, 1]) + 
        np.random.randn(len(cl_sample))*cr_error(cl_sample[:, 1], cl_sample[:, 0], cl_sample[:, 3])
    )
    cl_sample = cl_sample[cl_sample[:, 1]>cr_min]
    cl_sample = cl_sample[cl_sample[:, 1]<cr_max]

    cl_sample[:, 0] = cl_sample[:, 0] + np.random.randn(len(cl_sample))*z_err_model.predict(
        np.column_stack((
            cl_sample[:, 0],
            cl_sample[:, 2],
        ))
    )

    cl_sample = cl_sample[cl_sample[:, 0]>zm_min]
    cl_sample = cl_sample[cl_sample[:, 0]<zm_max]

    return cl_sample
