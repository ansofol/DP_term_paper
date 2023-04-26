import numpy as np 
import tools as tools 


def EGM_DC(t,sol,par): 
    for i in range(par.Ntypes):
        
        # Highly handhold way to deal with different types 
        if i <=1: 
            theta = par.theta_high 
        else: 
            theta = par.theta_low
            
        if t >= par.S_max - 1: # In this case the individual will work next period 
            
            for eps in range(par.neps): #Calculates the expected value and expected marginal utility if working in the next period
                # Should be vectorized 
                ell_next = tools.interp_linear_1d(par.a_grid,sol.ell[i,t+1,1,t+1,:,eps],par.a_grid)
                
                m_next = (1+r)*par.a_grid + ell_next*wage_func(t+1,t+1,theta,par.eps_grid[eps],par) 
                
                V_next_work += tools.interp_linear_1d(sol.m,sol.V[i,t+1,1,t+1,:,eps],m_next)*par.eps_w[eps]

                c_next =  tools.interp_linear_1d(sol.m,sol.c[i,t+1,1,t+1,:,eps],m_next)

                E_margu_work += marg_u(c, par)*par.eps_w[eps]

            c = (par.beta*(1+par.r)*E_margu_work)**(1/-par.rho)
            m = par.a_grid + c
            V = util(c,par) + par.beta*V_next_work

            sol.c[i,t,0,t] = c
            sol.m[i,t,0,t] = m 
            sol.V[i,t,0,t] = V
            
        else: # Can continue to study
    
            #1) Find expected value and expected marginal utility if working in next period  
            for eps in range(par.neps):
                
                ell_next = tools.interp_linear_1d(par.a_grid,sol.ell[i,t+1,1,t+1,:,eps],par.a_grid)
                
                m_next = (1+r)*par.a_grid + ell_next*wage_func(t+1,t+1,theta,par.eps_grid[eps],par) 
                
                V_next_work += tools.interp_linear_1d(sol.m,sol.V[i,t+1,1,t+1,:,eps],m_next)*par.eps_w[eps]

                c_next =  tools.interp_linear_1d(sol.m,sol.c[i,t+1,1,t+1,:,eps],m_next)

                E_margu_work += marg_u(c, par)*par.eps_w[eps]  

            #2) Find value, consumption and marginal utility if studying in next period

            m_next = (1+par.r)*par.a_grid + transfer(i,par) 

            V_next_study = tools.interp_linear_1d(sol.m[i,t+1,0,t+1,:,0],sol.V[i,t+1,0,t+1,:,0],m_next)

            c_next_study = tools.interp_linear_1d(sol.m[i,t+1,0,t+1,:,0],sol.c[i,t+1,0,t+1,:,0],m_next) 

            margu_study = marg_u(c_next_study,par) 
            
            # Choice probabilities 
            V = np.array([V_next_study,V_next_work])

            V_max = V.max(axis=0) 

            log_sum = V_max + np.log((np.sum(np.exp(V-V_max),axis=0)))

            ccp = np.exp(V-log_sum)

            # Expected marginal utility 

            E_margu = ccp[0]*margu_study+ccp[1]*E_margu_work

            c_now = (par.beta*(1+par.r)*E_margu)**(1/-par.rho)
            m_now = par.a_grid + c_now 
            V_now = util(c_now,par) + par.beta*log_sum

            # Upper Envelope 
            m = sorted(m_now) 
            I = m_now 
            c = [x for _,x in sorted(zip(I,c_now))]
            V = [x for _,x in sorted(zip(I,V_now))]

            for  i in range(np.size(m_now)-2): 
                m_low = m_now[i]
                m_high = m_now[i+1]
                c_slope = (c_now[i+1]-c_now[i])/(m_high-m_low)

                for j in range(len(m)): 
                    if m[j]>=m_low and m[j]<=m_high: 
                        c_geuss = c_now[i] + c_slope*(m[j]-m_low)
                        V_geuss = value_of_choice_study(m[j], c_geuss, i, t, sol, par)

                        if v_geuss >V[j]: 
                            V[j] = V_geuss 
                            c[j] = c_geuss 
            
            sol.c[i,t,0,t] = c
            sol.m[i,t,0,t] = m 
            sol.V[i,t,0,t] = V



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
    c**-par.rho

def value_of_choice_study(m,c,type,t,sol,par): 
    a = m-c 

    for eps in range(par.neps):
                    
        ell_next = tools.interp_linear_1d(par.a_grid,sol.ell[type,t+1,1,t+1,:,eps],a)
        
        m_next = par.a_grid + ell_next*wage_func(t,t,theta,par.eps_grid[eps],par) 
        
        V_next_work += tools.interp_linear_1d(sol.m,sol.V[type,t+1,1,t+1,:,eps],a)*par.eps_w[eps]

    #2) Find value, consumption and marginal utility if studying in next period

    m_next = (1+par.r)*par.a_grid + transfer(i,par)

    V_next_study = tools.interp_linear_1d(sol.m[type,t+1,0,t+1],sol.V[type,t+1,0,t+1],m_next)
    
    # Choice probabilities 
    V = np.array([V_next_study,V_next_work])

    V_max = V.max(axis=0) 

    log_sum = V_max + np.log((np.sum(np.exp(V-V_max),axis=0)))

    return util(c,par) + par.beta*log_sum


