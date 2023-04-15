import numpy as np 
import tools as tools


def EGM(s,d,sol,par): 

    m =  np.zeros((2,par.s_max,par.Na))
    c =  np.zeros((2,par.s_max,par.Na))
    v =  np.zeros((2,par.s_max,par.Na))

    for educ in range(s):
    # raw EGM 
        if d == 1: # Choose to work 
            m_next = par.R*par.grid_a[s,:] + par.income_work[educ] # Next period cash on hand  
            
            v_next = tools.interp_linear_1d(sol.m[s+1,d,educ,par.N_bottom:],sol.v[s+1,d,educ,par.N_bottom:], m_next) 
            c_next = tools.interp_linear_1d(sol.m[s+1,d,educ,par.N_bottom:],sol.c[s+1,d,educ,par.N_bottom:], m_next)

            E_marg_util = marg_util(c_next, par)

            c[d,educ] = par.R*par.beta*(1/E_marg_util) # Using inverse expected marginal utility to find consumption in current period 

            m[d,educ] = c[d,educ] + par.grid_a[s,:] 

            v[d,educ] = util(c[d,educ],par) + par.beta*v_next
            
            
        else: 
            pass 
    print(m[1,educ,0])
    return m,c,v
# Solve conditional on choice 

# Get raw values 

# Find the secoundary upper envelope (conditional on status - if working not needed)


#def studying 

#def working 






def util(c,par): 
    return np.log(c) 

def marg_util(c,par): 
    return 1/c 

#def logsum 