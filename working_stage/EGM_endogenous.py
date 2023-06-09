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
            #  current wage 
            
            EMU = 0
            # loop over epsilon shocks tomorrow
            for ii_eps,  eps_plus in enumerate(par.eps_grid):

                # next period policy function
                ## intermezzo: identify part of grid that is positive
                _m_next_grid = sol.m[i_type, t+1, 1,i_S,:,ii_eps] # next period beginning of state assets
                _c_next_grid = sol.c[i_type, t+1,1,i_S,:,ii_eps] # next period consumption

                m_next_grid = _m_next_grid[_m_next_grid>0]
                c_next_grid = _c_next_grid[_m_next_grid>0]

                # append zero assets
                m_next_grid = np.append([0], m_next_grid)
                m_size = m_next_grid.size


                c0 = tools.interp_linear_1d_scalar(_m_next_grid[m_size+1:], _c_next_grid[m_size+1:], 0)
                c_next_grid = np.append([c0], c_next_grid)

                # moving on the well-known part
                m_next = (1+par.r)*a
                try:
                    c_interp = tools.interp_linear_1d_scalar(m_next_grid, c_next_grid, m_next)
                except:
                    print(i_type,t,1,i_S,i_a,i_eps)
                MU = marginal_util(c_interp)
                EMU += MU*par.eps_w[ii_eps]
            
            # invert marginal utility
            c_endo[i_a] = inv_marginal_util(par.beta*(1+par.r)*EMU)
            
            # compute labor from intratemporal FOC
            ell_endo[i_a] = (wage*c_endo[i_a]**(-par.rho))**(1/par.nu)

            # endogenous grid
            m_endo[i_a] = a - wage*ell_endo[i_a] + c_endo[i_a]

            if m_endo[i_a] < 0:
                print(i_type, t, 1, i_S, i_a, i_eps)

            #c_exo = tools.interp_linear_1d(m_endo, c_endo, (1+par.r)*par.a_grid)
            #ell_exo = tools.interp_linear_1d(m_endo, ell_endo, (1+par.r)*par.a_grid)
        
        sol.c[i_type, t, 1, i_S, par.Ba:, i_eps] = c_endo #tools.interp_linear_1d(m_endo, c_endo, (1+par.r)*par.a_grid)
        sol.ell[i_type, t, 1, i_S, par.Ba:, i_eps] = ell_endo #tools.interp_linear_1d(m_endo, ell_endo, (1+par.r)*par.a_grid)
        sol.m[i_type, t, 1, i_S, par.Ba:, i_eps] = m_endo

        # interpolate back to exogenous grid
        #c_interp = interpolate.RegularGridInterpolator([m_endo], c_endo, method='linear', bounds_error=False, fill_value=None)
        #c_exo = c_interp((1+par.r)*par.a_grid)

        #ell_interp = interpolate.RegularGridInterpolator([m_endo], ell_endo, method='linear', bounds_error=False, fill_value=None)
        #ell_exo = ell_interp((1+par.r)*par.a_grid)
        #a_exo = (1+par.r)*par.a_grid + wage_vec*ell_exo - c_exo
  
        # If budget constraint binds at positive level of assets, add points under budget constraint
        m_min = m_endo.min()
        if m_min > 0: # if budget constraint binds at positive assets
            m_bc = tools.nonlinspace(1e-8, m_min, par.Ba+1, par.a_phi)[:-1]
            sol.m[i_type, t, 1, i_S, :par.Ba, i_eps] = m_bc
            for i_m, m in enumerate(m_bc):
                # intra-period budget FOC must hold
                intra_FOC = lambda c: m + wage*((wage*c**(-par.rho))**(1/par.nu)) - c
                root = optimize.root_scalar(intra_FOC, bracket=(1e-12, 20000), x0=a+1e-12)
                assert root.converged
                c = root.root
                sol.c[i_type, t, 1, i_S, i_m, i_eps] = c
                sol.ell[i_type, t, 1, i_S, i_m, i_eps] = (wage*c**(-par.rho))**(1/par.nu)



        # check budget constraint
        #for i_a, a in enumerate(par.a_grid):
        #    if a_exo[i_a] < 0:
        #        a_exo[i_a] = 0

        #        wage = wage_vec[i_a]

                # ensure intra-period FOC holds
        #        intra_FOC = lambda c: a + wage*((wage*c**(-par.rho))**(1/par.nu)) - c
        #        root = optimize.root_scalar(intra_FOC, bracket=(1e-12, 20000), x0=a+1e-12)
        #        assert root.converged
        #        c_exo[i_a] = root.root
        #        ell_exo[i_a] = (wage*c_exo[i_a]**(-par.rho))**(1/par.nu)

        #assert np.all(a_exo >=0)

        # interpolate back to exogenous grids (I don't know if this is the way)
        #sol.c[i_type, t, 1, i_S, :, i_eps] = c_exo #tools.interp_linear_1d(m_endo, c_endo, (1+par.r)*par.a_grid)
        #sol.ell[i_type, t, 1, i_S, :, i_eps] = ell_exo #tools.interp_linear_1d(m_endo, ell_endo, (1+par.r)*par.a_grid)
        #sol.a[i_type, t, 1, i_S, :, i_eps] = a_exo

        # compute value function
        #for i_a, a in enumerate(par.a_grid):
        #    v_next_vec = sol.V[i_type, t+1, 1, i_S, :, :]
        #    EV_next = v_next_vec@par.eps_w
        #    v_next_interp = interpolate.RegularGridInterpolator([par.a_grid], EV_next, 
        #                                                        method='linear', bounds_error=False, fill_value=None)
        #    a_next = (1+par.r)*a_exo[i_a]
        #    v_next = v_next_interp([a_next])
        #    sol.V[i_type, t, 1, i_S, i_a, :] = v_next


        #sol.c[i_type, t, 1, i_S, :, i_eps] = c_endo
        #sol.ell[i_type, t, 1, i_S, :, i_eps] = ell_endo
        #sol.m[i_type, t, 1, i_S, :, i_eps] =  m_endo
                    
