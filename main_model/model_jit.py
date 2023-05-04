import numpy as np
import sys
from scipy import optimize
from types import SimpleNamespace
import tools as tools
from DC_EGM import EGM_DC
import EGM
from consav import newton_raphson
from numba import njit, jit

class Model():

    def __init__(self):
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace()
        self.sim = SimpleNamespace()

    def setup(self):

        par = self.par

        # types
        # there are four types :)
        par.Ntypes = 4

        # cognitive types
        par.theta_high = 2
        par.theta_low = 1.5

        # transfer types
        par.phi_high = 1
        par.phi_low = 0.1

        # combined
        par.theta = np.array([par.theta_high, par.theta_high, par.theta_low, par.theta_low])
        par.phi = np.array([par.phi_high, par.phi_low, par.phi_high, par.phi_low])

        # preferences
        par.rho = 2. # CRRA coefficient
        par.nu = 1. # inverse frisch
        par.beta = 0.98

        # Extreme value type one distribution 
        par.sigma_taste = 1

        # education
        par.Smax = 4

        # income
        par.sigma = 1 # or something
        # maybe education specific age profile here
        par.r = 1/par.beta - 1

        # time
        par.Tmax = 10

        # grids
        par.a_phi = 1.1
        par.a_min = 1e-16
        par.a_max = 1000
        par.Na = 200
        par.Ba = 10

        par.neps = 5

    def set_grids(self):
        
        par = self.par
        sol = self.sol

        #### grids ###
        # assets
        par.a_grid = tools.nonlinspace(par.a_min, par.a_max, par.Na, par.a_phi)

        #### education ####
        par.S_grid = np.arange(par.Smax+1)
        par.lambda_vec = (np.arange(par.Smax+1)+1)*0.1

        #### productivity shocks ####
        par.eps_grid, par.eps_w = tools.gauss_hermite(par.neps) 

        #### solution grids ####
        shape = (par.Ntypes, par.Tmax, 2, par.Smax+1, par.Na + par.Ba, par.neps)
        sol.c = np.zeros(shape) + np.nan
        sol.ell = np.zeros(shape) + np.nan
        sol.ccp_work = np.zeros(shape) + np.nan
        sol.V = np.zeros(shape) + np.nan
        sol.m = np.zeros(shape) + np.nan
        sol.a = np.zeros(shape) + np.nan

    def solve(self):
        par = self.par
        sol = self.sol

        
        for t in range(par.Tmax-1, -1, -1):
            for i_type in range(par.Ntypes):
                for i_S, S in enumerate(par.S_grid):

                    if t == par.Tmax-1: # solve last period one state at a time
                        # parrallelized solve last function here
                        for i_a,a in enumerate(par.a_grid):
                            for i_eps, eps in enumerate(par.eps_grid):
                                idx =(i_type,t,1,i_S,par.Ba+i_a,i_eps)
                                
                                # leave no assets
                                #obj = lambda x: -self.util_last(x,wage,a)
                                wage = np.array([self.wage_func(i_S,t,i_type,eps)])
                                beta = np.array([par.beta])
                                rho = np.array([par.rho])
                                nu = np.array([par.nu])
                                r = np.array([par.r])
                                print(a, wage, beta, rho, nu, r)
                                res = solve_last_v(np.array([a]), wage, beta, rho, nu, r)

                                ell = res[1]
                                #c = wage*ell + a
                                c = res[0]

                                sol.c[idx] = c
                                sol.ell[idx] = ell
                                sol.a[idx] = 0
                                sol.m[idx] = a
                                sol.V[idx] = self.util_work(c, ell)

                    else:
                        EGM.EGM_step(t, i_type, i_S, self) # EGM in working stage
                    if t < par.Smax:
                        EGM_DC(i_type, t, sol, par)


    #####################
    # Utilities
    #####################

    
    
    def wage_func(self, i_S, t, i_type, eta):
        par = self.par
        return np.exp(par.lambda_vec[i_S]*np.log(par.theta[i_type]) + eta)

    def util_work(self,c,ell):
        par = self.par

        # impose penalty if lower bound is violated
        penalty = 0
        penalty += -1000*np.fmin(ell, 0)
        penalty += -1000*np.fmin(c,0)

        return (c**(1-par.rho))/(1-par.rho) - (ell**(1+par.nu))/(1+par.nu) - penalty
    

    def util_last(self, ell, w,a):
        c = ell*w+a
        return self.util_work(c, ell)
    
    def marginal_util(self, c):
        par = self.par
        return c**-par.rho
    
    def inv_marginal_util(self, m):
        par = self.par
        return m**(-1/par.rho)
    
        
    def retire_value(self, a):
        par = self.par
        v = np.zeros(a.shape)
        v[a >= 0] = (a[a>=0]**0.5)
        v[a<0] = 0
        return v
    
    def util_last_v(self, c, ell, w, a):
        par = self.par

        u = self.util_work(c, ell)
        m_next = (a + w*ell- c)*(1+par.r)

        # penalty for violating budget constraint
        penalty = np.zeros(m_next.shape)
        penalty -= 100*np.fmin(0, m_next)

        return u + par.beta*self.retire_value(m_next) - penalty

    # solve last period
    #def solve_last_v(self, idx):
    #    par = self.par
    #    i_type,t,_,i_S,i_a,i_eps = idx
    #    wage = self.wage_func(i_S,t,i_type,par.eps_grid[i_eps])
    #    print(wage)
    #    a = par.a_grid[i_a-par.Ba] #- par.Ba to adjust for bottom grid points in solution grids

    #    obj = lambda x: -self.util_last_v(x[0], x[1], wage, a)

    #    res = optimize.minimize(obj, x0=(a,1),method='nelder-mead', options={'maxiter':200})
        #assert res.success
    #    return res
    
    def exp_MU(self, i_type,t,i_work,i_S,i_a):
        """
        Expected marginal utility in period t (quadrature over epsilon shocks)
        """
        par = self.par
        sol = self.sol
        EMU = 0
        for ii_eps, eps in enumerate(par.eps_grid):
            if i_work == 1: 
                m_next_grid = sol.m[i_type, t, 1,i_S,par.Ba:,ii_eps] # next period beginning of state assets
                c_next_grid = sol.c[i_type, t,1,i_S,par.Ba:,ii_eps] # next period consumption
            else:
                m_next_grid = sol.m[i_type, t,0,i_S,:,ii_eps] # next period beginning of state assets
                c_next_grid = sol.c[i_type, t,0,i_S,:,ii_eps] # next period consumption

            m_next = (1+par.r)*par.a_grid[i_a]
            c_interp = tools.interp_linear_1d_scalar(m_next_grid, c_next_grid, m_next)
            MU = self.marginal_util(c_interp)
            EMU += MU*par.eps_w[ii_eps]
        return EMU

def util(c,par): 
    return c**(1-par.rho)/(1-par.rho)

#@njit
#def obj_last(ell, a, w, rho, nu):
#    c = ell*w+a
#    u = (c**(1-rho))/(1-rho) - (ell**(1+nu))/(1+nu)
#    return -u[0]

# solve last period
#@njit
#def solve_last(a, w, rho, nu):
#    res = newton_raphson.optimizer(obj_last, x0=a, args=(a,w,rho,nu))
#    return res

@njit
def obj_last_v(x, a, w, beta, rho, nu, r):
    c = x[0]
    ell = x[1]
    m_next = (1+r)*(a-c+w*ell)
    if m_next > 0:
        bonus = m_next**0.5 #retirement value here
    else:
        bonus = 100*m_next
    u = (c**(1-rho))/(1-rho) - (ell**(1+nu))/(1+nu) + bonus
    return -u[0]

@njit
def solve_last_v(a, w, beta, rho, nu, r):
    print(a, w, beta, rho, nu, r)
    x0 = np.array([a[0]/3, 1/w[0]])
    res = newton_raphson.optimizer(obj_last_v, x0=x0, args=(a,w,beta,rho,nu,r))
    return res
