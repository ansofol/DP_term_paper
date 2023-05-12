import numpy as  np 
import tools as tools 
import model as model 
from DC_EGM import transfer
from DC_EGM import value_of_choice_study

wage_func = model.wage_func

def simulate(sim,sol,par): 

    #Initialize 
    sim.m[:,0] = par.m_initial

    random = par.random


    sim.type = random.choice(par.Ntypes,par.N,replace=True,p=par.dist)

    u_choice = random.rand(par.N,par.Smax) # Used for education choices 

    wage_shock = random.choice(par.neps,(par.N,par.Tsim),replace=True,p = par.eps_w/np.sum(par.eps_w))

    
    for t in range(par.Tsim): 
        for i in range(par.N):
            shock = wage_shock[i,t]
            choice = 0
            type = sim.type[i]

            if t <= par.Smax -1: # Periodes involving discrete decision for students
                # Find choices for people who study 
                if sim.S[i,t] == t: 
                    
                    m_study = sim.m[i,t] + transfer(type,par)
                    V_study = tools.interp_linear_1d_scalar(sol.m[type,t,0,t,:,0], sol.V[type,t,0,t,:,0], m_study)

                    m_work =sim.m[i,t] 
                    V_work = tools.interp_linear_1d_scalar(sol.m[type,t,1,t,par.Ba:,shock], sol.V[type,t,1,t,par.Ba:,shock], m_work)

                    V = np.array([V_study,V_work])

                    p = ccp(V,par)

                    choice = u_choice[i,t] < 1 - p # Choice to continue to study
                    sim.S[i,t+1] += sim.S[i,t] + choice # Change Education status

                # Find Income and consumption dependent on choice 
            if sim.S[i,t] == t and t < par.Smax and choice == 1 : # People who continue to study 
                income = sim.m[i,t] + transfer(type,par)
                sim.c[i,t] = tools.interp_linear_1d_scalar(sol.m[type,t,0,t,:,0], sol.c[type,t,0,t,:,0], sim.m[i,t] + transfer(type,par))
                sim.m[i,t+1] = (1+par.r)*(income - sim.c[i,t])

            else:  # People who work  
                s = int(np.max(sim.S[i,:],axis=0))
                sim.c[i,t] = tools.interp_linear_1d_scalar(sol.m[type,t,1,s,par.Ba:,shock], sol.c[type,t,1,s,par.Ba:,shock], sim.m[i,t])
                sim.ell[i,t] = tools.interp_linear_1d_scalar(sol.m[type,t,1,s,par.Ba:,shock], sol.ell[type,t,1,s,par.Ba:,shock], sim.m[i,t])
                sim.S[i,t] = sim.S[i, t-1]
                income = sim.m[i,t] + wage_func(s,t,type,par.eps_grid[shock],par)*sim.ell[i,t]
                if t < par.Tmax - 1:
                    sim.m[i,t+1] = (1+par.r)*(income - sim.c[i,t])
                        

def ccp(V,par): 
    V_max = V.max(axis=0) 

    log_sum = V_max + par.sigma_taste*np.log((np.sum(np.exp((V-V_max)/par.sigma_taste),axis=0)))

    ccp = np.exp((V-log_sum)/par.sigma_taste)

    return ccp[1]

"""
def wage_func(i_S, t, i_type, eta, par):
    return np.exp(par.lambda_vec[i_S]*np.log(1+par.theta[i_type]) + eta)
"""


