import numpy as np
import sys
sys.path.append('../main_model')
from model import Model
import EGM
from scipy import optimize

class WorkModel(Model):
    pass

    def solve(self):
        par = self.par
        sol = self.sol

        
        for t in range(par.Tmax-1, -1, -1):
            for i_type in range(par.Ntypes):
                for i_S, S in enumerate(par.S_grid):
                    for i_a,a in enumerate(par.a_grid):
                        for i_eps, eps in enumerate(par.eps_grid):
                            idx =(i_type,t,1,i_S,i_a,i_eps)
                    
                            if t == par.Tmax-1:
                                # leave no assets
                                #obj = lambda x: -self.util_last(x,wage,a)
                                wage = self.wage_func(i_S,t,i_type,eps)
                                res = self.solve_last(idx)
                                
                                assert res.success

                                if res.success:
                                    ell = res.x
                                    c = wage*ell + a

                                    sol.c[idx] = c
                                    sol.ell[idx] = ell
                                    print('success')
                            else:
                                # egm step
                                pass

                    #EGM step             
                    wage_grid = self.wage_func(i_S, t, i_type, par.eps_grid)
                    EGM.EGM_step(t, par, wage_grid, sol)

                                


                                
                    

        # egm step
    def solve_last(self, idx):
        par = self.par
        i_type,t,_,i_S,i_a,i_eps = idx
        wage = self.wage_func(i_S,t,i_type,par.eps_grid[i_eps])
        a = par.a_grid[i_a]

        print(wage)

        obj = lambda x: -self.util_last(x,wage,a)

        res = optimize.minimize(obj, x0=1e-8,method='nelder-mead')
        return res
   

    def solve_last_wrong(self, i_type):

        par = self.par
        sol = self.sol

        

        for i_S, S in enumerate(par.S_grid):
            wage_grid = self.wage_func(i_S, -1, par.theta[i_type],par.eps_grid)

            solved = False

            for i_a, a in enumerate(par.a_grid):
                for i_eps, eps in enumerate(par.eps_grid):

                    idx = (i_type, -1, 1, i_S, i_a, i_eps)
                    
                    # objective function
                    obj = lambda x: -self.value_of_choice_last(x[0], x[1], a, wage_grid[i_eps])

                    # borrowing constraint
                    #bc = lambda x: (1+par.r)*a + wage_grid[i_S]*x[1] - x[0] 
                    #constraint = {'fun':bc, 'type':'ineq'}

                    bounds = ((0,np.inf),(0,np.inf))

                    # initial guess
                    if solved:
                        c0 = c
                        ell0 = ell
                    else:
                        c0 = a/2 + 1e-8 # consume half of assets
                        ell0 = wage_grid[i_eps]*c0 # work to finance consumption


                    res = optimize.minimize(obj, (1,1), 
                                            method='nelder-mead', 
                                            bounds=bounds)
                    print(res)

                    if res.success:
                        solved = True
                        sol.c[idx] = c = res.x[0]
                        sol.ell[idx] = ell = res.x[1]
                    #sol.a[idx] = a - sol.c[idx]
                    
                    


            
            

    def wage_func(self, i_S, t, i_type, eta):
        par = self.par
        return np.exp(par.lambda_vec[i_S]*np.log(par.theta[i_type]) + eta)

    def util_work(self,c,ell):
        par = self.par
        return (c**(1-par.rho))/(1-par.rho) - (ell**(1+par.nu))/(1+par.nu)
    
    def util_last(self, ell, w,a):
        c = ell*w+a

        penalty = 0
        if ell < 0:
            penalty += abs(ell)*1000
        return np.array(self.util_work(c, ell))
    
    def marginal_util(self, c):
        par = self.par
        return c**-par.rho
    
    def inv_marginal_util(self, m):
        par = self.par
        return m**(-1/par.rho)
    
        
    def retire_value(self, a):
        par = self.par
        return a**0.5 #fill out something better here
    

