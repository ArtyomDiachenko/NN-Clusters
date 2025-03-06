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
    batchsize=2**16,
)

m[:] = np.load("catalogue_sampler/selection_function_fit/model_parameters.npy")

scaler_min, scaler_scale = np.load("catalogue_sampler/selection_function_fit/scaler_parameters.npy")
scaler = MinMaxScaler()
scaler.min_ = scaler_min
scaler.scale_ = scaler_scale

def selection_function(x):
    x = scaler.transform(x)
    return m.predict(x)[0]

area_array = np.load("catalogue_sampler/data/effective_area/effective_area.npy")
zlim_array = np.load("catalogue_sampler/data/effective_area/z_array.npy")

df = pd.read_csv("catalogue_sampler/data/skymap/skykoords.csv")
df = df[["Texp", "nh", "weight"]]

sky_map_exp = np.array(df["Texp"])
sky_map_nh = np.array(df["nh"])
sky_map_weight = np.array(df["weight"])

del df

true_pars = {}
with open('catalogue_sampler/true_cosmo_pars.txt', 'r') as f:
    for line in f:
        line = line.split()
        true_pars[line[0]] = float(line[1])

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

params_true = {
            'flat': True, 'H0': true_pars['h0'], 'Om0': true_pars['omega_m'], 
            'Ob0': true_pars['ob'], 'sigma8': true_pars['sigma_8'], 
            'ns': true_pars['ns'], 'Tcmb0': 2.7255, 'Neff': 3.046
}
cosmo_true = cosmology.setCosmology('true_cosmo', **params_true, persistence = '')

p_cr = [
    true_pars['a_cr'], true_pars['b_cr'], true_pars['delta_cr'],
    true_pars['gamma_cr'], true_pars['sigma_cr']
]
p_l = [
    true_pars['a_l'], true_pars['b_l'], true_pars['delta_l'],
    true_pars['gamma_l'], true_pars['sigma_l']
]

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
    

z_err_model = joblib.load("catalogue_sampler/data/error_model/redshift_error_model.joblib")

n_cells = 1024

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
    lnm_list, 
    np.random.normal(lncr_list, p_cr[-1]),
    np.random.normal(lnl_list, p_l[-1])
))

cl_sample = cl_sample[cl_sample[:, 3]>-3]
cl_sample = cl_sample[cl_sample[:, 2]>-6.9]

idxs = np.random.choice(
    len(sky_map_weight), size=len(cl_sample), p=sky_map_weight
)

cl_sample = np.column_stack((
    cl_sample,
    sky_map_exp[idxs],
    sky_map_nh[idxs],
    sky_map_weight[idxs]
))

prob = selection_function(np.column_stack((
    cl_sample[:, 2],
    np.log(cl_sample[:, 4]),
    np.log(cl_sample[:, 5]),
    cl_sample[:, 0]
)))

prob = np.random.binomial(1, prob)
cl_sample = cl_sample[prob.reshape(-1)==1]
cl_sample[:, 3] = truncnorm.rvs(a=0, b=np.inf, loc=np.exp(cl_sample[:, 3]), scale=np.exp(cl_sample[:, 3]))
cl_sample = cl_sample[cl_sample[:, 3]>lambda_min]
cl_sample = cl_sample[cl_sample[:, 3]<lambda_max]

cl_sample[:, 2] = np.exp(cl_sample[:, 2]) + np.random.randn(len(cl_sample))*cr_error(cl_sample[:, 2], cl_sample[:, 0], cl_sample[:, 4])
cl_sample = cl_sample[cl_sample[:, 2]>cr_min]
cl_sample = cl_sample[cl_sample[:, 2]<cr_max]

cl_sample[:, 0] = cl_sample[:, 0] + np.random.randn(len(cl_sample))*z_err_model.predict(
    np.column_stack((
        cl_sample[:, 0],
        cl_sample[:, 3],
    ))
)

cl_sample = cl_sample[cl_sample[:, 0]>zm_min]
cl_sample = cl_sample[cl_sample[:, 0]<zm_max]

np.save('cl_sample', cl_sample)
