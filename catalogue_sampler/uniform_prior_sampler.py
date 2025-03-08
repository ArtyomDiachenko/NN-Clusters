import numpy as np

omega_m_lim = [0.1, 0.4]
sigma_8_lim = [0.5, 1.1]

ob_lim = [0.042, 0.049]
h0_lim = [50, 90]
ns_lim = [0.92, 1.0]

a_cr_lim = [0, 0.3]
b_cr_lim = [0, 3]
delta_cr_lim = [-3, 3]
gamma_cr_lim = [-3, 3]
sigma_cr_lim = [0.05, 1.5]

a_l_lim = [1, 80]
b_l_lim = [0, 3]
delta_l_lim = [-3, 3]
gamma_l_lim = [-3, 3]
sigma_l_lim = [0.05, 1.5]

r_cr_l_lim = [-0.9, 0.9]

def uniform_prior():
    pars = {}
    pars['omega_m'] = np.random.uniform(omega_m_lim[0], omega_m_lim[1])
    pars['sigma_8'] = np.random.uniform(sigma_8_lim[0], sigma_8_lim[1])
    pars['h0'] = np.random.uniform(h0_lim[0], h0_lim[1])
    pars['ns'] = np.random.uniform(ns_lim[0], ns_lim[1])
    pars['ob'] = np.random.uniform(ob_lim[0], ob_lim[1])
    pars['a_cr'] = np.random.uniform(a_cr_lim[0], a_cr_lim[1])
    pars['b_cr'] = np.random.uniform(b_cr_lim[0], b_cr_lim[1])
    pars['delta_cr'] = np.random.uniform(delta_cr_lim[0], delta_cr_lim[1])
    pars['gamma_cr'] = np.random.uniform(gamma_cr_lim[0], gamma_cr_lim[1])
    pars['sigma_cr'] = np.random.uniform(sigma_cr_lim[0], sigma_cr_lim[1])
    pars['a_l'] = np.random.uniform(a_l_lim[0], a_l_lim[1])
    pars['b_l'] = np.random.uniform(b_l_lim[0], b_l_lim[1])
    pars['delta_l'] = np.random.uniform(delta_l_lim[0], delta_l_lim[1])
    pars['gamma_l'] = np.random.uniform(gamma_l_lim[0], gamma_l_lim[1])
    pars['sigma_l'] = np.random.uniform(sigma_l_lim[0], sigma_l_lim[1])
    pars['r_cr_l'] = np.random.uniform(r_cr_l_lim[0], r_cr_l_lim[1])
    return pars
    