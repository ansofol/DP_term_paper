import tools
import numpy as np
from scipy import optimize, interpolate
from model import wage_func

def EGM_step(t,i_type,i_S,model):
    
    par = model.par
    sol = model.sol
    marginal_util = model.marginal_util
    inv_marginal_util = model.inv_marginal_util

    
    for i_eps,eps in enumerate(par.eps_grid): 

        # allocate space to store edogenous grids
        c_endo = np.zeros(par.Na) + np.nan
        ell_endo = np.zeros(par.Na) + np.nan
        m_endo = np.zeros(par.Na) + np.nan
        wage = wage_func(i_S,t,i_type,eps, par)

        # expected marginal utility in next period by end of period assets
        EMU = model.exp_MU(i_type,t+1,1,i_S,par.a_grid)
        c_endo = inv_marginal_util(par.beta*(1+par.r)*EMU) # consumption from Euler
        ell_endo = ell_from_FOC(c_endo, wage, par) # labor from intra period FOC
        m_endo = par.a_grid - wage*ell_endo + c_endo # endogenous grid


        # interpolate back to exogenous grid
        c_exo = tools.interp_linear_1d(m_endo, c_endo, par.a_grid)
        ell_exo = tools.interp_linear_1d(m_endo, ell_endo, par.a_grid)
        a_exo = par.a_grid + wage*ell_exo - c_exo
  
        # check budget constraint
        for i_a, a in enumerate(par.a_grid):
            if a_exo[i_a] < 0:
                a_exo[i_a] = 0

                # ensure intra-period FOC holds
                intra_FOC = lambda c: a + wage*ell_from_FOC(c, wage, par) - c 
                root = optimize.root_scalar(intra_FOC, bracket=(1e-12, 20000), x0=a+1e-12) #maybe this bracket should be adjusted to something a bit more general
                assert root.converged
                c_exo[i_a] = root.root
                ell_exo[i_a] = ell_from_FOC(root.root, wage, par) 

        assert np.all(a_exo >=0)

        # interpolate back to exogenous grids 
        sol.c[i_type, t, 1, i_S, par.Ba:, i_eps] = c_exo 
        sol.ell[i_type, t, 1, i_S, par.Ba:, i_eps] = ell_exo 
        sol.a[i_type, t, 1, i_S, par.Ba:, i_eps] = a_exo
        sol.m[i_type, t, 1, i_S, par.Ba:, i_eps] = par.a_grid

        # used for computing Euler errors
        sol.EMU[i_type, t, 1, i_S, par.Ba:, i_eps] = EMU

    # compute value function
    v_next_vec = sol.V[i_type, t+1, 1, i_S, par.Ba:, :]
    EV_next = v_next_vec@par.eps_w
    m_next = a_exo*(1+par.r) #### OBS: changed this from par.a_grid bc I forgot i had changed what the grids meant
    v_next = tools.interp_linear_1d(sol.m[i_type, t+1, 1, i_S, par.Ba:, i_eps], EV_next, m_next)
    util = c_exo**(1-par.rho)/(1-par.rho) - par.vartheta*ell_exo**(1+par.nu)/(1+par.nu) + par.beta*v_next
    sol.V[i_type, t, 1, i_S, par.Ba:, :] = np.repeat(util, par.neps).reshape(par.Na, par.neps)


def ell_from_FOC(c, wage, par):
    return ((wage/par.vartheta)*c**-par.rho)**(1/par.nu)
