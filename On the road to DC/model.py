
from types import SimpleNamespace
import numpy as np 
import EGM as egm 
import tools as tools

class study_model(): 
    def __init__(self,full_model = False): 
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace()
        self.sim = SimpleNamespace() 

        par = self.par 
        
        # Grids 
        par.s_max = 10 # Time
        par.Na = 200
        par.a_max = 100
        par.a_phi =1.1
        
        #Hvad pointen med den her ??
        par.N_bottom = 10 # Number of points at the bottom in the G2-EGM algorithm

        # Income and preference parameters 
        par.R = 1.02
        par.Su = 1 
        par.beta = 0.96
        par.rho = 1.5

        if full_model == False: 
            par.income_work = [x*(par.s_max-s) for x,s in zip(1+np.arange(par.s_max), np.arange(par.s_max))]

        self.set_grids()
       

    def set_grids(self): 
        par = self.par 

        par.grid_a = np.nan + np.zeros([par.s_max,par.Na])
        for s in range (par.s_max): 
            par.grid_a[s,:]= tools.nonlinspace(0+1e-16, par.a_max, par.Na, par.a_phi) 
    
    def solve(self): 
        # Unpack
        par = self.par 
        sol = self.sol 

        shape = (par.s_max,2,par.s_max,par.Na+par.N_bottom)

        sol.m = np.nan + np.zeros(shape)
        sol.c = np.nan + np.zeros(shape) 
        sol.v = np.nan + np.zeros(shape) 

        # Need to handle savings when changing however for now treat as entire life 
        # Consume everything 
        sol.m[par.s_max-1,:,:,:] = tools.nonlinspace(0+1e-16, par.a_max, par.Na + par.N_bottom, par.a_phi) 
        sol.c[par.s_max-1,:,:,:] = tools.nonlinspace(0+1e-16, par.a_max, par.Na + par.N_bottom, par.a_phi) 
        sol.v[par.s_max-1,:,:,:] = egm.util(sol.c[par.s_max-1,:,:,:],par)

        for s in range(par.s_max-2,-1,-1): 
            m,c,v = egm.EGM(s,1,sol,par)

            for educ in range(s):
                m_con = np.linspace(0+1e-16,m[1,educ,0],par.N_bottom)
                c_con = m_con
                v_con = egm.util(c_con,par) + par.beta*tools.interp_linear_1d(sol.m[s+1,1,educ,par.N_bottom:],sol.v[s+1,1,educ,par.N_bottom:], m_con)
                
                m_temp = m[1,educ,:]
                c_temp = c[1,educ,:]
                v_temp = v[1,educ,:]


                sol.m[s,1,educ] = np.append(m_con,m_temp)
                sol.c[s,1,educ] = np.append(c_con,c_temp)
                sol.v[s,1,educ] = np.append(v_con,v_temp)



            #for d in range(2): # There are some timing issues I think I havne't thougha about 
             #   m,c,v = egm.EGM(s,d,sol,par)

              #  m_con = np.linspace(0,m[0],par.N_bottm)
               # c_con = m_con 
                #v_con = value_of_choice(m_con,c_con,d,s,sol,par)

                #sol.m[s,d] = np.append(m_con,m)
                #sol.c[s,d] = np.append(c_con,c)
                #sol.v[s,d] = np.append(v_con,v)


     





        
        




        
         
