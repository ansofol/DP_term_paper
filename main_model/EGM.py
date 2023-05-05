import tools
import numpy as np
from scipy import optimize, interpolate

def EGM_step(t,i_type,i_S,model):
    
    par = model.par
    sol = model.sol
    wage_func = model.wage_func
    marginal_util = model.marginal_util
    inv_marginal_util = model.inv_marginal_util

    
    for i_eps,eps in enumerate(par.eps_grid): 

        # allocate space to store edogenous grids
        c_endo = np.zeros(par.Na) + np.nan
        ell_endo = np.zeros(par.Na) + np.nan
        m_endo = np.zeros(par.Na) + np.nan
        wage = wage_func(i_S,t,i_type,eps)
        for i_a, a in enumerate(par.a_grid): # loop over end of period assets

            EMU = model.exp_MU(i_type,t+1,1,i_S,i_a)
            
            # invert marginal utility
            c_endo[i_a] = inv_marginal_util(par.beta*(1+par.r)*EMU)
            
            # compute labor from intratemporal FOC
            ell_endo[i_a] = ell_from_FOC(c_endo[i_a], wage, par) 

            # endogenous grid
            m_endo[i_a] = a - wage*ell_endo[i_a] + c_endo[i_a]
        

        # interpolate back to exogenous grid
        c_interp = interpolate.RegularGridInterpolator([m_endo], c_endo, method='linear', bounds_error=False, fill_value=None)
        c_exo = c_interp(par.a_grid)

        ell_interp = interpolate.RegularGridInterpolator([m_endo], ell_endo, method='linear', bounds_error=False, fill_value=None)
        ell_exo = ell_interp(par.a_grid)
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

        # compute value function
        for i_a, a in enumerate(par.a_grid):
            v_next_vec = sol.V[i_type, t+1, 1, i_S, par.Ba:, :]
            EV_next = v_next_vec@par.eps_w
            v_next_interp = interpolate.RegularGridInterpolator([sol.m[i_type, t, 1, i_S, par.Ba:, i_eps]], EV_next, 
                                                                method='linear', bounds_error=False, fill_value=None)
            m_next = a
            v_next = v_next_interp([m_next])
            sol.V[i_type, t, 1, i_S, par.Ba+i_a, :] = model.util_work(c_exo[i_a], ell_exo[i_a]) + par.beta*v_next


def ell_from_FOC(c, wage, par):
    return ((wage/par.vartheta)*c**-par.rho)**(1/par.nu)
