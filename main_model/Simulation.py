import numpy as  np 
import tools as tools 
import model as model 
from DC_EGM import transfer
from DC_EGM import value_of_choice_study


def simulate(sim,sol,par): 

    #Initialize 
    sim.a[:,:,0] = par.a_initial
    
    u_choice = np.random.rand(par.Ntypes,par.N,par.Smax) # Used for education choices 
    
    wage_shock = np.random.choice(par.neps,(par.Ntypes,par.N,par.Tsim),replace=True,p = par.eps_w/np.sum(par.eps_w))

    
    for t in range(par.Tsim): 
        for z in range(par.Ntypes):
            for i in range(par.N):
                shock = wage_shock[z,i,t]
                choice = 0

                if t <= par.Smax -1: # Periodes involving discrete decision for students
                    # Find choices for people who study 
                    if sim.S[z,i,t] == t: 
                        
                        m_study = sim.a[z,i,t] + transfer(z,par)
                        V_study = tools.interp_linear_1d_scalar(sol.m[z,t,0,t,:,0], sol.V[z,t,0,t,:,0], m_study)

                        m_work =sim.a[z,i,t] 
                        V_work = tools.interp_linear_1d_scalar(sol.m[z,t,1,t,par.Ba:,shock], sol.V[z,t,1,t,par.Ba:,shock], m_work)

                        V = np.array([V_study,V_work])

                        p = ccp(V,par)

                        choice = u_choice[z,i,t] < 1 - p # Choice to continue to study
                        sim.S[z,i,t+1] += sim.S[z,i,t] + choice # Change Education status

                    # Find Income and consumption dependent on choice 
                if sim.S[z,i,t] == t and t < par.Smax and choice == 1 : # People who continue to study 
                    income = sim.a[z,i,t] + transfer(z,par)
                    sim.c[z,i,t] = tools.interp_linear_1d_scalar(sol.m[z,t,0,t,:,0], sol.c[z,t,0,t,:,0], sim.a[z,i,t] + transfer(z,par))
                    sim.a[z,i,t+1] = (1+par.r)*(income - sim.c[z,i,t])

                else:  # People who work  
                    s = int(np.max(sim.S[z,i,:],axis=0))
                    sim.c[z,i,t] = tools.interp_linear_1d_scalar(sol.m[z,t,1,s,par.Ba:,shock], sol.c[z,t,1,s,par.Ba:,shock], sim.a[z,i,t])
                    sim.ell[z,i,t] = tools.interp_linear_1d_scalar(sol.m[z,t,1,s,par.Ba:,shock], sol.ell[z,t,1,s,par.Ba:,shock], sim.a[z,i,t])
                    income = sim.a[z,i,t] + wage_func(s,t,z,par.eps_grid[shock],par)*sim.ell[z,i,t]
                    if t < par.Tmax - 1:
                        sim.a[z,i,t+1] = (1+par.r)*(income - sim.c[z,i,t])
                        

def ccp(V,par): 
    V_max = V.max(axis=0) 

    log_sum = V_max + par.sigma_taste*np.log((np.sum(np.exp((V-V_max)/par.sigma_taste),axis=0)))

    ccp = np.exp((V-log_sum)/par.sigma_taste)

    return ccp[1]


def wage_func(i_S, t, i_type, eta, par):
    return np.exp(par.lambda_vec[i_S]*np.log(par.theta[i_type]) + eta)



