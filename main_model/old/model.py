import numpy as np
import scipy.optimize
from types import SimpleNamespace
import tools

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
        par.rho = 2 # CRRA coefficient
        par.nu = 1 # inverse frisch
        par.beta = 0.9

        # education
        par.Smax = 5

        # income
        par.sigma = 1 # or something
        # maybe education specific age profile here
        par.r = 1/par.beta - 1

        # time
        par.Tmax = 45

        # grids
        par.a_phi = 1.1
        par.a_min = 0.0
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
        sol.m = np.zeros(shape)
        sol.a = np.zeros(shape) + np.nan
        

