import numpy as  np 
from project_code import tools 
from project_code import model 
from project_code.DC_EGM import transfer
from project_code.DC_EGM import value_of_choice_study

wage_func = model.wage_func

def simulate(sim,sol,par):

    sim.m[:,0] = 1

    random = par.random
    sim.type = random.choice(par.Ntypes,par.N,replace=True,p=par.dist)
    u_choice = random.rand(par.N,par.Smax) # Used for education choices 
    wage_shock = random.choice(par.neps,(par.N,par.Tsim),replace=True,p = par.eps_w)
    sim.wage_shock = wage_shock

    for t in range(par.Tsim): 

        shock = wage_shock[:,t]
        choice = np.zeros(par.N)
        work = np.ones(par.N).astype(bool) # indicator for working
        edu = np.max(sim.S, axis=1).astype(int)
        sim.wage[:,t] = wage_func(edu, t, sim.type, par.eps_grid[shock], par)

        m_study = sim.m[:,t] + par.phi[sim.type]
        m_work = sim.m[:,t]

        for type in range(par.Ntypes): # loop over four different types
            sample = (sim.type == type)

            if t <= par.Smax - 1: # in period where some may still be studying
                sub_sample = sample & (edu == t)

                V_study = tools.interp_linear_1d(sol.m[type,t,0,t,:,0], sol.V[type,t,0,t,:,0], m_study[sub_sample])
                
                EV_next = sol.V[type,t,1,t,par.Ba:,:]@par.eps_w
                EV_work = tools.interp_linear_1d(sol.m[type,t,1,t,par.Ba:,0], EV_next, m_work[sub_sample])
                # V_work = tools.interp_2d_vec(sol.m[type,t,1,t,par.Ba:,0], par.eps_grid, sol.V[type,t,1,t,par.Ba:,:], m_work[sub_sample], par.eps_grid[shock[sub_sample]])

                V = np.array([V_study, EV_work])
                p = ccp(V,par)
                choice[sub_sample] = u_choice[sub_sample,t] < 1-p # Choice to continue to study
                sim.S[sub_sample,t+1] += sim.S[sub_sample,t] + choice[sub_sample] # Change Education status

                # find income and consumption depending on choice: studying
                study = (choice == 1) & sub_sample # people who continue to study
                work[study] = False # set working indicator to zero
                sim.c[study, t] = tools.interp_linear_1d(sol.m[type,t,0,t,:,0], sol.c[type,t,0,t,:,0], m_study[study])
                sim.m[study, t+1] = (1+par.r)*(m_study[study] - sim.c[study, t])


            # find income and consumption depending on choice: working
            for s_now in range(par.Smax+1): # loop through different edu levels
                sample_work = work & (sim.type==type) & (edu==s_now)

                sim.c[sample_work, t] = tools.interp_2d_vec(sol.m[type,t,1,s_now,par.Ba:,0], 
                                                            par.eps_grid, sol.c[type,t,1,s_now,par.Ba:,:], 
                                                            m_work[sample_work], 
                                                            par.eps_grid[shock[sample_work]])
                
                sim.ell[sample_work, t] = tools.interp_2d_vec(sol.m[type,t,1,s_now,par.Ba:,0], 
                                                              par.eps_grid, sol.ell[type,t,1,s_now,par.Ba:,:], 
                                                              m_work[sample_work], 
                                                              par.eps_grid[shock[sample_work]])
                
                income = m_work[sample_work] + sim.wage[sample_work, t]*sim.ell[sample_work,t]

                if t < par.Tmax - 1:
                    sim.m[sample_work,t+1] = (1+par.r)*(income - sim.c[sample_work,t])





def simulate_old(sim,sol,par): 

    #Initialize 
    sim.m[:,0] = par.m_initial

    random = par.random


    sim.type = random.choice(par.Ntypes,par.N,replace=True,p=par.dist)

    u_choice = random.rand(par.N,par.Smax) # Used for education choices 

    wage_shock = random.choice(par.neps,(par.N,par.Tsim),replace=True,p = par.eps_w/np.sum(par.eps_w))
    sim.wage_shock = wage_shock

    
    for t in range(par.Tsim): 
        for i in range(par.N):
            shock = wage_shock[i,t]
            choice = 0
            type = sim.type[i]
            
            edu = int(np.max(sim.S[i]))
            sim.wage[i,t] = wage_func(edu, t, type, par.eps_grid[shock], par)

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


