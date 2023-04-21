import tools
import numpy as np

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

        for i_a, a in enumerate(par.a_grid): # loop over end of period assets
            #  current wage 
            wage = wage_func(i_S,t,i_type,eps)
        
            EMU = 0
            # loop over epsilon shocks tomorrow
            for ii_eps,  eps_plus in enumerate(par.eps_grid):

                # next period policy function
                m_next_grid = sol.m[i_type, t+1, 1,i_S,:,ii_eps] # next period beginning of state assets
                c_next_grid = sol.c[i_type, t+1,1,i_S,:,ii_eps] # next period consumption
                m_next = (1+par.r)*a
                try:
                    c_interp = tools.interp_linear_1d_scalar(m_next_grid, c_next_grid, m_next)
                except:
                    print(i_type,t,1,i_S,i_a,i_eps)
                MU = marginal_util(c_interp)
                EMU += MU*par.eps_w[ii_eps]
            
            # invert marginal utility
            c_now = inv_marginal_util(par.beta*(1+par.r)*EMU)
            
            # compute labor from intratemporal FOC
            ell_now = (wage*c_now**(-par.rho))**(1/par.nu)

            # endogenous grid
            m_now = a - wage*ell_now + c_now

            # check borrowing constraint

            # store solutions
            c_endo[i_a] = c_now
            ell_endo[i_a] = ell_now
            m_endo[i_a] = m_now

        # interpolate back to exogenous grids (I don't know if this is the way)
        #sol.c[i_type, t, 1, i_S, :, i_eps] = tools.interp_linear_1d(m_endo, c_endo, (1+par.r)*par.a_grid)
        #sol.ell[i_type, t, 1, i_S, :, i_eps] = tools.interp_linear_1d(m_endo, ell_endo, (1+par.r)*par.a_grid)

        sol.c[i_type, t, 1, i_S, :, i_eps] = c_endo
        sol.ell[i_type, t, 1, i_S, :, i_eps] = ell_endo
        sol.m[i_type, t, 1, i_S, :, i_eps] =  m_endo
                    
