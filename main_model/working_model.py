import numpy as np
import sys
sys.path.append('../main_model')
from model import Model
import EGM
from scipy import optimize
import tools

class WorkModel(Model):
    pass

    def solve(self):
        par = self.par
        sol = self.sol

        
        for t in range(par.Tmax-1, -1, -1):
            for i_type in range(par.Ntypes):
                for i_S, S in enumerate(par.S_grid):

                    if t == par.Tmax-1: # solve last period one state at a time
                        for i_a,a in enumerate(par.a_grid):
                            for i_eps, eps in enumerate(par.eps_grid):
                                idx =(i_type,t,1,i_S,par.Ba+i_a,i_eps)
                                
                                # leave no assets
                                #obj = lambda x: -self.util_last(x,wage,a)
                                wage = self.wage_func(i_S,t,i_type,eps)
                                res = self.solve_last_v(idx)

                                assert res.success

                                if res.success:
                                    c, ell = res.x
                                    #c = wage*ell + a

                                    sol.c[idx] = c
                                    sol.ell[idx] = ell
                                    sol.a[idx] = 0
                                    sol.m[idx] = a
                                    sol.V[idx] = self.util_work(c, ell)
                                else:
                                    print(f'Did not converge at {idx}')
                                    # this becomes an issue if we allow for borrowing.
                                    #we can maybe try some trouble shooting or different starting values - or we can just interpolate over the holes in the policy functions :))
                    else:
                        EGM.EGM_step(t, i_type, i_S, self)
                        
                        

                                


                                
                    

    # solve last period
    def solve_last(self, idx):
        par = self.par
        i_type,t,_,i_S,i_a,i_eps = idx
        wage = self.wage_func(i_S,t,i_type,par.eps_grid[i_eps])
        a = par.a_grid[i_a-par.Ba] #- par.Ba to adjust for bottom grid points in solution grids

        obj = lambda x: -self.util_last(x,wage,a)

        res = optimize.minimize(obj, x0=1,method='nelder-mead', options={'maxiter':200})
        #assert res.success
        return res
    
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
            
            

    def wage_func(self, i_S, t, i_type, eta):
        par = self.par
        return np.exp(par.lambda_vec[i_S]*np.log(par.theta[i_type]) + eta)

    def util_work(self,c,ell):
        par = self.par

        # impose penalty if lower bound is violated
        penalty = 0
        #if ell <= 0:
        penalty += -1000*np.fmin(ell, 0)
        #print('ell penalized')
        #if c <= 0:
        #print('c penalized')
        penalty += -1000*np.fmin(c,0)

        return (c**(1-par.rho))/(1-par.rho) - (ell**(1+par.nu))/(1+par.nu) - penalty
    

    def util_last(self, ell, w,a):
        c = ell*w+a
        return np.array(self.util_work(c, ell))
    
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
    def solve_last_v(self, idx):
        par = self.par
        i_type,t,_,i_S,i_a,i_eps = idx
        wage = self.wage_func(i_S,t,i_type,par.eps_grid[i_eps])
        a = par.a_grid[i_a-par.Ba] #- par.Ba to adjust for bottom grid points in solution grids

        obj = lambda x: -self.util_last_v(x[0], x[1], wage, a)

        res = optimize.minimize(obj, x0=(a,1),method='nelder-mead', options={'maxiter':200})
        #assert res.success
        return res
