import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import mass_function
import os
import tqdm
import emcee
from multiprocessing import Pool
from matplotlib import pyplot as plt
from scipy.integrate import dblquad
from scipy.interpolate import interpn
from scipy.signal import correlate2d
import corner

os.environ["OMP_NUM_THREADS"] = "1"

true_pars = {}
with open('m_z_sampler/true_cosmo_pars.txt', 'r') as f:
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

solid_angle = 13_000 * (np.pi/180)**2

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
    )*3000*solid_angle
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


n_cells = 1024

n_points = n_cells + 1

z_a = np.linspace(z_min, z_max, n_points)
lnm_a = np.linspace(lnM_min, lnM_max, n_points)
z_grid, lnm_grid = np.meshgrid(z_a, lnm_a, indexing='ij')
dz = z_a[1] - z_a[0]
dlnm = lnm_a[1] - lnm_a[0]

dn_dmdz_grid = np.zeros((n_points, n_points))
for i in range(n_points):
    dn_dmdz_grid[i] = dn_dmdz(lnm_a, z_a[i], cosmo_true)

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

np.save('cl_sample', cl_sample)


# nwalkers = 64
# tau = 40
# init = [0.3, 0.8, 70, 0.95, 0.045] + np.random.randn(nwalkers, 5)*0.001
# with Pool() as pool:
#     sampler = emcee.EnsembleSampler(nwalkers, 5, mz_likelihood, pool=pool)
#     sampler.run_mcmc(init, 6_000, progress=True)

# cosmo_chain = sampler.get_chain(discard=1000, flat=True)
# blob = sampler.get_blobs(discard=1000, flat=True)
# comb_chain = np.column_stack((cosmo_chain, blob))

# corner.corner(
#     comb_chain, 
#     labels=['$\Omega_{m}$', '$\sigma_{8}$', 'H', 'ns', '$\Omega_{b}$', 'N'],
#     show_titles=True,
#     truths=[
#         true_pars['omega_m'], true_pars['sigma_8'], 
#         true_pars['h0'], true_pars['ns'], true_pars['ob'],
#         len(cl_sample)
#     ]
# );
# plt.savefig('check.png')