import numpy as np
    
def exp_MU_vector(self, i_type,t, i_work, i_S):
    par = self.par
    sol = self.sol

    EMU = np.zeros(par.Na)

    if i_work == 1:
        m_next_grid = sol.m[i_type, t, 1,i_S,par.Ba:] # next period beginning of state assets
        c_next_grid = sol.c[i_type, t,1,i_S,par.Ba:] # next period consumption
    else:
        m_next_grid = sol.m[i_type, t,0,i_S,:,] # next period beginning of state assets
        c_next_grid = sol.c[i_type, t,0,i_S,:,] # next period consumption

    m_next = np.repeat((1+par.r)*par.a_grid, par.neps).reshape(par.Na, par.neps)
    m_next_test = (1+par.r)*par.a_grid

    print()
