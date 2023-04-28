import numpy as np 
import tools as tools 

def EGM(t,sol,par): # Help EGM - not to be used in the model
    for i in range(par.Ntypes):
        for s in range(par.Smax+1):

            if i <=1: 
                theta = par.theta_high 
            else: 
                theta = par.theta_low

            V_next_work = np.zeros((par.Na))
            E_margu_work = np.zeros((par.Na))


            for eps in range(par.neps): #Calculates the expected value and expected marginal utility if working in the next period
            # Should be vectorized 
                m_next = (1+par.r)*par.a_grid 
                    
                V_next_work += tools.interp_linear_1d(sol.m[i,t+1,1,s,:,eps],sol.V[i,t+1,1,s,:,eps],m_next)*par.eps_w[eps]

                c_next =  tools.interp_linear_1d(sol.m[i,t+1,1,s,:,eps],sol.c[i,t+1,1,s,:,eps],m_next)

                E_margu_work += marg_u(c_next, par)*par.eps_w[eps]

            
            c = (par.beta*(1+par.r)*E_margu_work)**(1/-par.rho)
            m = (par.a_grid + c)[:,np.newaxis] - par.eps_grid
            V = util(c,par) + par.beta*V_next_work

            sol.c[i,t,1,s,:,:] = c[:, np.newaxis]
            sol.m[i,t,1,s,:,:] = m 
            sol.V[i,t,1,s,:,:] = V[:, np.newaxis] 

        

def EGM_DC(t,sol,par): 
    for i in range(par.Ntypes):
        
        # Highly handhold way to deal with different types 
        if i <=1: 
            theta = par.theta_high 
        else: 
            theta = par.theta_low
         
        V_next_work = np.zeros((par.Na))
        E_margu_work = np.zeros((par.Na))

        if t >= par.Smax - 1: # In this case the individual will work next period 
           
            for eps in range(par.neps): #Calculates the expected value and expected marginal utility if working in the next period
                # Should be vectorized 
                m_next = (1+par.r)*par.a_grid 
                
                V_next_work += tools.interp_linear_1d(sol.m[i,t+1,1,t+1,:,eps],sol.V[i,t+1,1,t+1,:,eps],m_next)*par.eps_w[eps]

                c_next =  tools.interp_linear_1d(sol.m[i,t+1,1,t+1,:,eps],sol.c[i,t+1,1,t+1,:,eps],m_next)

                E_margu_work += marg_u(c_next, par)*par.eps_w[eps]

            c = (par.beta*(1+par.r)*E_margu_work)**(1/-par.rho)
            m = par.a_grid + c
            V = util(c,par) + par.beta*V_next_work



            sol.c[i,t,0,t,:,0] = c
            sol.m[i,t,0,t,:,0] = m 
            sol.V[i,t,0,t,:,0] = V
            
        else: # Can continue to study
    
            #1) Find expected value and expected marginal utility if working in next period  
            for eps in range(par.neps): #Calculates the expected value and expected marginal utility if working in the next period
                # Should be vectorized 
                m_next = (1+par.r)*par.a_grid 
                
                V_next_work += tools.interp_linear_1d(sol.m[i,t+1,1,t+1,:,eps],sol.V[i,t+1,1,t+1,:,eps],m_next)*par.eps_w[eps]

                c_next =  tools.interp_linear_1d(sol.m[i,t+1,1,t+1,:,eps],sol.c[i,t+1,1,t+1,:,eps],m_next)

                E_margu_work += marg_u(c_next, par)*par.eps_w[eps] 

            #2) Find value, consumption and marginal utility if studying in next period

            m_next = (1+par.r)*par.a_grid 

            V_next_study = tools.interp_linear_1d(sol.m[i,t+1,0,t+1,:,0],sol.V[i,t+1,0,t+1,:,0],m_next)

            c_next_study = tools.interp_linear_1d(sol.m[i,t+1,0,t+1,:,0],sol.c[i,t+1,0,t+1,:,0],m_next) 

            margu_study = marg_u(c_next_study,par) 
            #assert np.mean(np.isnan(margu_study)) ==0
            #assert np.mean(np.isnan(E_margu_work)) ==0

      
            
            # Choice probabilities 
            V = np.array([V_next_study,V_next_work])

            V_max = V.max(axis=0) 

            log_sum = V_max + par.sigma_taste*np.log((np.sum(np.exp((V-V_max)/par.sigma_taste),axis=0)))

            ccp = np.exp((V-log_sum)/par.sigma_taste)
            #assert np.mean(np.isnan(ccp))

            # Expected marginal utility 

            E_margu = ccp[0]*margu_study+ccp[1]*E_margu_work
            #assert np.mean(np.isnan(E_margu)) ==0
            c_now = (par.beta*(1+par.r)*E_margu)**(1/-par.rho)
            m_now = par.a_grid + c_now - transfer(i,par)
            V_now = util(c_now,par) + par.beta*log_sum
            #assert np.mean(np.isnan(c_now)) ==0 

            # Upper Envelope 
            m = sorted(m_now) 
            I = m_now 
            c = [x for _,x in sorted(zip(I,c_now))]
            V = [x for _,x in sorted(zip(I,V_now))]

            for  k in range(np.size(m_now)-2): # Go through "solution grid"
                m_low = m_now[k]
                m_high = m_now[k+1]
                c_slope = (c_now[k+1]-c_now[k])/(m_high-m_low)

                for j in range(len(m)): # Check values inside the sorted grid
                    if m[j]>=m_low and m[j]<=m_high: 
                        c_geuss = c_now[k] + c_slope*(m[j]-m_low) # interpolate consumption 
                        V_geuss = value_of_choice_study(m[j], c_geuss, i, t, sol, par)

                        if V_geuss >V[j]: # Check what consumption dominates
                            V[j] = V_geuss 
                            c[j] = c_geuss 
            
            sol.c[i,t,0,t,:,0] = c
            sol.m[i,t,0,t,:,0] = m 
            sol.V[i,t,0,t,:,0] = V



def transfer(type,par): 
    if type == 0 or type == 2: 
        return par.phi_high 
    if type ==1 or type ==3: 
        return par.phi_low

def util(c,par): 
    return c**(1-par.rho)/(1-par.rho)

def wage_func(i_S, t, theta, eta, par):
    return np.exp(par.lambda_vec[i_S]*np.log(theta) + eta)

def marg_u(c,par): 
    return c**(-par.rho)

def value_of_choice_study(m,c,type,t,sol,par): 
    a = m-c 

    V_next_work = np.nan

    for eps in range(par.neps):
        m_next = (1+par.r)*a
        V_next_work += tools.interp_linear_1d_scalar(sol.m[type,t+1,1,t+1,:,eps],sol.V[type,t+1,1,t+1,:,eps],m_next)*par.eps_w[eps]

    #2) Find value, consumption and marginal utility if studying in next period

    m_next = (1+par.r)*a

    V_next_study = tools.interp_linear_1d_scalar(sol.m[type,t+1,0,t+1,:,0],sol.V[type,t+1,0,t+1,:,0],m_next)
    
    # Choice probabilities 
    V = np.array([V_next_study,V_next_work])

    V_max = V.max(axis=0) 

    log_sum = V_max + par.sigma_taste*np.log((np.sum(np.exp((V-V_max)/par.sigma_taste),axis=0)))

    return util(c,par) + par.beta*log_sum


