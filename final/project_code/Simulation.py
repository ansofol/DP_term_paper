import numpy as  np 
from project_code import tools 
from project_code import model 
wage_func = model.wage_func

def simulate(sim,sol,par):
    """
    This function simulates the choices and outcomes of individuals over time based on a given set of
    parameters and a pre-existing solution.
    
    Args:
      sim: a simulation object containing arrays for variables to be simulated
      sol: A dictionary containing the solution to the dynamic programming problem, including the value
    function, consumption function, and labor supply function.
      par: A dictionary containing various model parameters such as the number of types, the number of
    periods, the interest rate, the wage distribution, and the distribution of taste shocks.
    """

    sim.m[:,0] = 1

    random = par.random #random number generator

    # draw types, wage shocks and uniform variable to be used for taste shocks
    sim.type = random.choice(par.Ntypes,par.N,replace=True,p=par.dist)
    u_choice = random.rand(par.N,par.Smax) # Used for education choices 
    wage_shock = random.choice(par.neps,(par.N,par.Tsim),replace=True,p = par.eps_w)
    sim.wage_shock = wage_shock

    for t in range(par.Tsim): 

        # unpack
        shock = wage_shock[:,t]
        choice = np.zeros(par.N)
        work = np.ones(par.N).astype(bool) 
        edu = np.max(sim.S, axis=1).astype(int)
        m_study = sim.m[:,t] + par.phi[sim.type]
        m_work = sim.m[:,t]

        sim.wage[:,t] = wage_func(edu, t, sim.type, par.eps_grid[shock], par)

        for type in range(par.Ntypes): # loop over four different types
            sample = (sim.type == type)

            # in period where some may still be studying
            if t <= par.Smax - 1:
                sub_sample = sample & (edu == t)

                # Value of studying
                V_study = tools.interp_linear_1d(sol.m[type,t,0,t,:,0], 
                                                 sol.V[type,t,0,t,:,0], 
                                                 m_study[sub_sample])
                
                # Expected value of working
                EV_next = sol.V[type,t,1,t,par.Ba:,:]@par.eps_w
                EV_work = tools.interp_linear_1d(sol.m[type,t,1,t,par.Ba:,0], 
                                                 EV_next, m_work[sub_sample])
                
                # Compute choice probabilities and draw study choice
                V = np.array([V_study, EV_work])
                p = ccp(V,par)
                choice[sub_sample] = u_choice[sub_sample,t] < 1-p # Choice to continue to study
                sim.S[sub_sample,t+1] += sim.S[sub_sample,t] + choice[sub_sample] # Change Education status

                #### Choice conditional on studying ###
                study = (choice == 1) & sub_sample # people who continue to study
                work[study] = False # set working indicator to zero

                # interpolate from consumption solution
                sim.c[study, t] = tools.interp_linear_1d(sol.m[type,t,0,t,:,0], 
                                                         sol.c[type,t,0,t,:,0], 
                                                         m_study[study])
                
                # asset transition
                sim.m[study, t+1] = (1+par.r)*(m_study[study] - sim.c[study, t])


            ### Choice conditional on working ###
            for s_now in range(par.Smax+1): # loop through different edu levels
                sample_work = work & (sim.type==type) & (edu==s_now)

                # interpolate from consumption solution
                sim.c[sample_work, t] = tools.interp_2d_vec(sol.m[type,t,1,s_now,par.Ba:,0], 
                                                            par.eps_grid, sol.c[type,t,1,s_now,par.Ba:,:], 
                                                            m_work[sample_work], 
                                                            par.eps_grid[shock[sample_work]])
                
                # interpolate from labor solution
                sim.ell[sample_work, t] = tools.interp_2d_vec(sol.m[type,t,1,s_now,par.Ba:,0], 
                                                              par.eps_grid, sol.ell[type,t,1,s_now,par.Ba:,:], 
                                                              m_work[sample_work], 
                                                              par.eps_grid[shock[sample_work]])
                
                # asset transition
                income = m_work[sample_work] + sim.wage[sample_work, t]*sim.ell[sample_work,t]
                if t < par.Tmax - 1:
                    sim.m[sample_work,t+1] = (1+par.r)*(income - sim.c[sample_work,t])



def ccp(V,par): 
    """
    The function calculates conditional choice probabilities based on a given set of inputs.
    
    Args:
      V: V is a numpy array representing the utility values of studying vs. working.
      par: Object containing parameters, incl. scale of taste shock.
    
    Returns:
      the conditional choice probability (ccp) of working in given period.
    """
    
    V_max = V.max(axis=0) 
    log_sum = V_max + par.sigma_taste*np.log((np.sum(np.exp((V-V_max)/par.sigma_taste),axis=0)))
    ccp = np.exp((V-log_sum)/par.sigma_taste)
    return ccp[1]

