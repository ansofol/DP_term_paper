import numpy as np
from scipy import optimize
from types import SimpleNamespace
from  project_code import tools
from project_code.DC_EGM import EGM_DC
import project_code.EGM as EGM
from project_code.auxiliary_funcs import *

class Model():

    def __init__(self):
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace()
        self.sim = SimpleNamespace()

    def setup(self):
        """
        Begins setup
            - Sets model parameters
            - Sets parameters for solution grids and simulation
        """

        par = self.par
        par.easy_par = False # set parameters so last period is analytically solvable

        # types
        par.Ntypes = 4

        # cognitive types
        par.theta_high =  0.9
        par.theta_low = 0.1

        # transfer types
        par.phi_high = 5
        par.phi_low = 0.5

        # preferences
        par.rho = 1.5 # CRRA coefficient
        par.nu = 3 # inverse frisch
        par.beta = 0.975
        par.vartheta = 0.0415
        par.kappa = 1
        
        # scale of extreme value type one distribution 
        par.sigma_taste = 1

        # education
        par.Smax = 4
        par.lambda_max = 4

        # income
        par.sigma = 1 # or something
        par.r = 0.018 

        # time
        par.Tmax = 10

        # grids
        par.a_phi = 1.3
        par.a_min = 1e-5 #check this later
        par.a_max = 1000
        par.Na = 200
        par.Ba = 10
        par.neps = 5

        # Simulation 
        par.N = 10_000 # Number of individuals to simulate 
        par.Tsim = par.Tmax #Periods to simulate 
        par.m_initial = 3
        par.random = np.random 
        par.dist = [0.25, 0.25, 0.25, 0.25]

        #Estimation 
        par.Ns = 10


    def set_grids(self):
        """
        Completes setup and sets grids:
            - Type and state grids that depend on parameters specified in self.setup
            - Initiates solution grids
            - Initiates simulation grids
        """
        
        par = self.par
        sol = self.sol
        sim = self.sim

        # Update parameters if solution is analytical
        if par.easy_par:
            par.beta = 0.98
            par.r = 1/par.beta - 1
            par.kappa = 1
            par.rho = 1
            par.nu = 1 

        #### grids ###
        # Types
        par.theta = np.array([par.theta_high, par.theta_high, par.theta_low, par.theta_low])
        par.phi = np.array([par.phi_high, par.phi_low, par.phi_high, par.phi_low])

        # assets
        par.a_grid = tools.nonlinspace(par.a_min, par.a_max, par.Na, par.a_phi)

        #### education ####
        par.S_grid = np.arange(par.Smax+1)
        par.lambda_vec = - tools.nonlinspace(-par.lambda_max, 0 , par.Smax+1, 1.03)
        par.lambda_vec = np.sort(par.lambda_vec)

        #### productivity shocks ####
        par.eps_grid, par.eps_w = tools.gauss_hermite(par.neps) 
        par.eps_w /= np.sqrt(np.pi)
        par.eps_grid *= np.sqrt(2)*par.sigma

        #### solution grids ####
        shape = (par.Ntypes, par.Tmax, 2, par.Smax+1, par.Na + par.Ba, par.neps)
        sol.c = np.zeros(shape) + np.nan
        sol.ell = np.zeros(shape) + np.nan
        sol.ccp_work = np.zeros(shape) + np.nan
        sol.V = np.zeros(shape) + np.nan
        sol.dV = np.zeros(shape) + np.nan
        sol.m = np.zeros(shape) + np.nan
        sol.a = np.zeros(shape) + np.nan
        sol.EMU = np.zeros(shape) + np.nan
        sol.adj_EMUell = np.zeros(shape) + np.nan

        ### Simulation grid ### 
        shape_sim = (par.N,par.Tsim)
        sim.c = np.zeros(shape_sim) 
        sim.S = np.zeros(shape_sim) 
        sim.ell = np.zeros(shape_sim) 
        sim.m = np.zeros(shape_sim) 
        sim.wage = np.zeros(shape_sim)
        sim.type = np.zeros(par.N)
        
        par.random.seed(1687) # Simulation seed

    def solve(self):
        """
        The function solves for optimal consumption and labor supply decisions using EGM and DC-EGM algorithm.

        Args:
         None
        Returns:
         None, modifies self.sol
        """
        par = self.par
        sol = self.sol

        # Loop over time, type, education, assets and income shocks
        for t in range(par.Tmax-1, -1, -1):
            for i_type in range(par.Ntypes):
                for i_S, S in enumerate(par.S_grid):

                    # Last period: solve using optimization
                    if t == par.Tmax-1: 
                        for i_a,a in enumerate(par.a_grid):
                            for i_eps, eps in enumerate(par.eps_grid):
                                idx =(i_type,t,1,i_S,par.Ba+i_a,i_eps)
                                
                                wage = wage_func(i_S,t,i_type,eps, par)
                                res = self.solve_last_v(idx)

                                if res.success:
                                    ell = res.x[1]
                                    c = res.x[0]

                                    sol.c[idx] = c
                                    sol.ell[idx] = ell
                                    sol.a[idx] = a + wage*ell - c
                                    sol.m[idx] = a
                                    sol.V[idx] = -res.fun
                                    sol.dV[idx] = -par.beta*par.kappa*(1+par.r)*((1+par.r)*(a+wage*ell-c))**(-par.rho)
                                else:
                                    print(f'Did not converge at {idx}')
                                    print(a)
                                    print(wage)
                                    assert res.success
                    else:
                        EGM.EGM_step(t, i_type, i_S, self) # EGM in working stage
                if t < par.Smax:
                    EGM_DC(i_type, t, sol, par) # DC_EGM in study stage
    

    def solve_study(self):
        ''' 
        Solves the study stage only, used to speed up estimation 
        Requires a full solution to working stage
        '''

        par = self.par 
        sol = self.sol 

        for t in range(par.Smax-1,-1,-1):
            for i_type in range(par.Ntypes): 
                EGM_DC(i_type,t,sol,par)

    
    
    ########################
    # Utilities
    ########################

    def util_work(self,c,ell):
        """
        The function calculates the flow utility of a given consumption and leisure inputs, based on 
        certain parameters.
        
        Args:
          c: consumption.
          ell: labor supply.
        
        Returns:
          flow utility value of given input.
        """
        par = self.par
        return (c**(1-par.rho))/(1-par.rho) - par.vartheta*(ell**(1+par.nu))/(1+par.nu)
    
    def marginal_util(self, c):
        """
        Calculates marginal utility of consumption.

        Args:
         c: consumption

        Returns:
          marginal utility.
        """
        par = self.par
        return c**-par.rho
    
    def inv_marginal_util(self, m):
        """
        Inverted marginal utility.

        Args:
          m: cash-on-hand
        
        Returns:
          inverse marginal utility of consumption.
        """
        par = self.par
        return m**(-1/par.rho)
    

    
    def exp_MU(self, i_type,t,i_work,i_S,a):
        """
        The function calculates the expected marginal utility of consumption for a given individual's type,
        time period, work status, state assets, using Gauss-Hermite quadrature over normally distributed
        income shocks.
        
        Args:
        i_type: type index
        t: time period
        i_work: Indicator for being in working stage.
        i_S: Years of education
        a: Current period beginning of period assets.
        
        Returns:
        the expected marginal utility (EMU) of consumption for a given individual type, time period, work
        status, current state assets, and shock realization.
        """
       
        par = self.par
        sol = self.sol
        EMU = 0
        for ii_eps, eps in enumerate(par.eps_grid):
            m_next_grid = sol.m[i_type, t, 1,i_S,i_work*par.Ba:,ii_eps] # next period beginning of state assets
            c_next_grid = sol.c[i_type, t,1,i_S,i_work*par.Ba:,ii_eps] # next period consumption
            m_next = a*(1+par.r)
            c_interp = tools.interp_linear_1d(m_next_grid, c_next_grid, m_next)
            MU = self.marginal_util(c_interp)
            EMU += MU*par.eps_w[ii_eps]
        return EMU
    

    def adj_exp_MUell(self, i_type,t,i_work,i_S,a):
        """
        This function calculates the wage adjusted expected marginal utility of labor supply for a given
        individual type, time period, work status, state assets.
        
        Args:
          i_type: type index
          t: time period
          i_work: indicator for being in the working stage.
          i_S: Years of education
          a: Current period beginning of period assets.
        
        Returns:
          the wage adjusted expected marginal utility (adj_EMU) as a float value.
        """

        par = self.par
        sol = self.sol
        adj_EMU = 0
        for ii_eps, eps in enumerate(par.eps_grid):
            wage = wage_func(i_S, t, i_type, eps, par)
            m_next_grid = sol.m[i_type, t, 1,i_S,i_work*par.Ba:,ii_eps] # next period beginning of state assets
            ell_next_grid = sol.ell[i_type, t,1,i_S,i_work*par.Ba:,ii_eps] # next period consumption
            m_next = a*(1+par.r)
            ell_interp = tools.interp_linear_1d(m_next_grid, ell_next_grid, m_next)
            MU = ell_interp**par.nu
            adj_EMU += (MU*par.eps_w[ii_eps])/wage
        return adj_EMU
    

    ########################
    # Last period solution
    ########################
    
    def last_util(self, x, a, wage):
        """
        The function calculates the period T utility of consumption, leisure, and retirement savings for a 
        given set of parameters.
        
        Args:
          x: A vector containing the current consumption level and labor supply level.
          a: current level of assets
          wage: wage.
        
        Returns:
          the utility from consumption, leisure, and retirement savings, given the current state variables
        (x, a, wage) and the model parameters (par). 
        """
        par = self.par

        c = x[0]
        ell =  x[1]
        a_next = (a + wage*ell - c)*(1+par.r)

        if par.rho == 1:
            uc = np.log(c)
            retire = np.log(a_next)
        else:
            uc = (1/(1-par.rho))*c**(1-par.rho)
            retire = (1/(1-par.rho))*a_next**(1-par.rho)

        dul = par.vartheta*(1/(1+par.nu))*ell**(1+par.nu)

        return  uc - dul + par.beta*par.kappa*retire


    def jac_last(self, x, a, wage):
        """
        The function calculates the Jacobian matrix of a two-dimensional system of equations related to
        consumption and labor supply in the last period.
        
        Args:
          x: a tuple containing the values of consumption (c) and labor supply (ell)
          a: Beginning of state assets.
          wage: Wage
        
        Returns:
          Numpy array `jac` which contains the partial derivatives of the
          objective function with respect to the consumption `c` and the labor supply `ell`.
        """
        par= self.par
        c = x[0]
        ell =  x[1]

        dc = c**(-par.rho) -par.beta*(1+par.r)*par.kappa*((1+par.r)*(a+wage*ell-c))**(-par.rho)
        dl = -par.vartheta*ell**(par.nu) + par.beta*par.kappa*(1+par.r)*wage*((1+par.r)*(a+wage*ell-c))**(-par.rho)
        jac= np.array([dc, dl])
        return jac


    def solve_last_v(self, idx):
        """
        This function solves for the optimal consumption and leisure choices and resulting utility given a
        set of parameters and an index.
        
        Args:
          idx: A tuple containing information about the individual and time period being solved for.
        Specifically, it contains the individual's type, time period, education index (i_S), asset grid
        index (i_a), and wage shock index (i_eps).
        
        Returns:
          An object of type `Namespace` containing the optimal values of
        consumption and leisure (`x`), the value of the utility function (`fun`), and a boolean indicating
        whether the optimization was successful (`success`).
        """
        par = self.par
        i_type,t,_,i_S,i_a,i_eps = idx
        wage = wage_func(i_S,t,i_type,par.eps_grid[i_eps], par)
        a = par.a_grid[i_a-par.Ba] #- par.Ba to adjust for bottom grid points in solution grids

        # Solve analytically if possible
        if self.par.easy_par:
            c = (1/2*a + np.sqrt((0.5*a)*(0.5*a) + 2*wage*wage/par.vartheta))/2
            ell = wage/par.vartheta*1/c
            x = (c,ell)
            V = self.last_util(x, a,wage)

            res = SimpleNamespace()
            res.x = x
            res.fun = V
            res.success = True

        # Otherwise, perform optimization
        else:        
            obj = lambda x: -self.last_util(x, a,wage)
            obj_jac = lambda x: -self.jac_last(x, a, wage)

            def bc(x):
                c = x[0]
                ell = x[1]
                bc = a + wage*ell - c
                return bc
            constr = {'fun': bc, 'type':'ineq'}

            res = optimize.minimize(obj, x0=(a,1/wage), method='slsqp', jac=obj_jac, constraints=constr)
        return res
    



    


    #########################
    # Verification          #
    #########################
    def euler_errors(self, bc_limit = 0.002):
        """
        This function calculates the Euler errors for consumption and labor supply and checks the marginal
        rate of substitution for a given set of parameters.
        
        Args:
          bc_limit: The lower bound for end-of-period assets below which the budget constraint is considered
        to be binding.
        
        Returns:
          five lists: Delta_time_c, epsilon_time_c, Delta_time_ell, epsilon_time_ell, and epsilon_MRS.
          Delta_time_c:         Average consumption Euler errors.
          epsilon_time_c:       Average absolute relative consumption Euler errors.
          Delta_time_ell:       Average labor Euler eurrors.
          epsilon_time_el:      Average aboslute relative labor Euler errors.
          epsilon_MRS:          Average MRS error.
        """
        par = self.par
        sol = self.sol
        sim = self.sim

        c = sim.c[:,:-1]
        ell = sim.ell[:,:-1]
        wage_sim = sim.wage[:,:-1]
        c_next = sim.c[:,1:]
        s = sim.S.max(axis=1)

        # shape
        shape = c.shape

        sim.Delta_c = np.zeros(shape) + np.nan
        sim.epsilon_c = np.zeros(shape) + np.nan
        sim.Delta_ell = np.zeros(shape) + np.nan
        sim.epsilon_ell = np.zeros(shape) + np.nan

        # Check when budget constraint bounds
        end_of_period_assets = (sim.m/(1+par.r))[:,1:].reshape(shape)
        non_bc =(end_of_period_assets>bc_limit).reshape(shape)

        for type in range(par.Ntypes):
            for edu in range(par.Smax+1):
                index = (s==edu)*(sim.type == type)
                current_c = c[index]
                current_a = end_of_period_assets[index]
                if current_a.size > 0:
                    x=1

                for t in range(par.Tmax-1):
                    if t < edu:
                        EMU = sol.EMU[type, t, 0, t, par.Ba:, 0] 
                        a_grid = par.a_grid 
                    else:
                        EMU = sol.EMU[type, t, 1, edu, par.Ba:, 0]
                        a_grid = par.a_grid
                        
                        # Labor euler error - depends on wage shock 
                        adj_EMUell = sol.adj_EMUell[type, t, 1, edu, par.Ba:,0]
                        for i_eps,eps in enumerate(par.eps_grid):
                            index_l = index*(sim.wage_shock[:,t] == i_eps)
                            wage = wage_func(edu, t, type, eps, par)
                            adj_EMUell_interp = wage*tools.interp_linear_1d(a_grid, adj_EMUell, end_of_period_assets[index_l, t])
                            euler_error_ell = (par.beta*(1+par.r)*adj_EMUell_interp)**(1/par.nu) - ell[index_l,t]
                            sim.Delta_ell[index_l,t] = euler_error_ell
                            sim.epsilon_ell[index_l, t] = np.log10(np.abs(euler_error_ell)/ell[index_l,t])

                    # consumption euler - does not depend on wage shock
                    EMU_interp = tools.interp_linear_1d(a_grid, EMU, (current_a)[:,t])
                    euler_error = self.inv_marginal_util(par.beta*(1+par.r)*EMU_interp) - current_c[:,t]

                    # save euler errorsfor i,j
                    sim.Delta_c[index, t] = euler_error
                    sim.epsilon_c[index, t] = np.log10(np.abs(euler_error)/current_c[:,t])
                    

        # check MRS - holds at all times when working
        epsilon_MRS = (par.vartheta/wage_sim)*ell**par.nu - c**(-par.rho)

        # return avg. euler over time, avg. relative euler over time
        Delta_time_c = [sim.Delta_c[:,t][non_bc[:,t]].mean() for t in range(par.Tmax-1)]
        epsilon_time_c = [sim.epsilon_c[:,t][non_bc[:,t]].mean() for t in range(par.Tmax-1)]
        Delta_time_ell = [sim.Delta_ell[:,t][non_bc[:,t]].mean() for t in range(par.Tmax-1)]
        epsilon_time_ell = [sim.epsilon_ell[:,t][non_bc[:,t]].mean() for t in range(par.Tmax-1)]


        return Delta_time_c, epsilon_time_c, Delta_time_ell, epsilon_time_ell, epsilon_MRS


