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
import corner

os.environ["OMP_NUM_THREADS"] = "1"

true_pars = {}
with open('true_cosmo_pars.txt', 'r') as f:
    for line in f:
        line = line.split()
        true_pars[line[0]] = float(line[1])

m_max = 10**16
m_min = 10**12

lnM_max = np.log(m_max)
lnM_min = np.log(m_min)

z_min = 0.04
z_max = 1.3

solid_angle = 0.0426464389

params_true = {
            'flat': True, 'H0': true_pars['h0'], 'Om0': true_pars['omega_m'], 
            'Ob0': true_pars['ob'], 'sigma8': true_pars['sigma_8'], 
            'ns': true_pars['ns'], 'Tcmb0': 2.7255, 'Neff': 3.046
}
cosmo_true = cosmology.setCosmology('true_cosmo', **params_true, persistence = '')

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

z_a = np.linspace(z_min, z_max, 400)
lnm_a = np.linspace(lnM_min, lnM_max, 400)

def mz_likelihood(X):
    om, s8, h0, ns, ob = X
    if (0.1<om<0.5 and 0.6<s8<1.2 and
        0.92<ns<1 and 0.042<ob<0.049 and 50<h0<90):
        params = {
            'flat': True, 'H0': h0, 'Om0': om, 'Ob0': ob, 'sigma8': s8, 
            'ns': ns, 'Tcmb0': 2.7255, 'Neff': 3.046
        }
        cosmo = cosmology.setCosmology('myCosmo', **params, persistence = '')
        dn_dzdlnm_a = np.empty((len(z_a), len(lnm_a)))
        for i in range(len(z_a)):
            dn_dzdlnm_a[i] = dn_dmdz(lnm_a, z_a[i], cosmo)
        lndn = np.sum(np.log(interpn((z_a, lnm_a), dn_dzdlnm_a, cl_sample)))
        n = np.trapz(np.trapz(dn_dzdlnm_a, z_a, axis=0), lnm_a)
        return lndn - n, n
    else:
        return -np.inf, None
    
ntot = dblquad(dn_dmdz, z_min, z_max, lnM_min, lnM_max, args=(cosmo_true,))[0]

print(f'N_tot = {int(ntot)}\n')

N = 60_000
maxn = 2

stop = False
while not stop:
    Z = np.linspace(z_min, z_max, N+1)
    LNM = np.linspace(lnM_min, lnM_max, N+1)
    dlnm = (LNM[1]-LNM[0])
    dz = (Z[1]-Z[0])
    cl_sample = []
    for i in tqdm.tqdm(range(N-1, -1, -1), desc=f'N = {N}'):
        mf = dn_dmdz(LNM[:-1]+0.5*dlnm, Z[i]+0.5*dz, cosmo_true)*dlnm*dz
        mfs = np.random.poisson(mf)
        maxs = np.max(mfs)
        if maxs>maxn:
            print(f'{maxs} = max sample > max_n = {maxn}; N = {N}')
            N += 10_000
            break
        for j in range(len(mfs)):
            if mfs[j] ==1:
                cl_sample.append([Z[i]+0.5*dz, LNM[j]+0.5*dlnm])
            elif mfs[j]==2:
                lnms = np.random.uniform(LNM[j], LNM[j+1], 2)
                zs = np.random.uniform(Z[i], Z[i+1], 2)
                cl_sample.append([zs[0], lnms[0]])
                cl_sample.append([zs[1], lnms[1]])
    else:
        stop=True

cl_sample = np.array(cl_sample)
np.save('cl_sample', cl_sample)
print('\nsample generated!\n')

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