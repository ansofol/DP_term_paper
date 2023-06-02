import numpy as np 
from project_code import tools
from project_code.auxiliary_funcs import *


# Set up Function that Solves the study stage using DC-EGM
def EGM_DC(i,t,sol,par): 

    """
    Implements the endogenous grid method (EGM) with discrete choice for an economic model.

    Parameters:
        i (int): Type index.
        t (int): Time period index.
        sol (object): namespace containing solution grids and values.
        par (object): namespace containing model parameters.

    Returns:
        None, instead  modifies the sol namespace 
    """

    # Highly handhold way to deal with different types 
    if i <=1: 
        theta = par.theta_high 
    else: 
        theta = par.theta_low

    # Initialize     
    V_next_work = np.zeros((par.Na))
    E_margu_work = np.zeros((par.Na))

    if t >= par.Smax -1: # In this case the individual will work next period 
        
        for eps in range(par.neps): #Calculates the expected value and expected marginal utility if working in the next period
            
            # Find pre decision income in the next period
            m_next = (1+par.r)*par.a_grid 
            
            # Find the value in the next period given m_next by interpolation and weight with the wage shock weight, and add it to the weighted sum
            V_next_work += tools.interp_linear_1d(sol.m[i,t+1,1,t+1,par.Ba:,eps],sol.V[i,t+1,1,t+1,par.Ba:,eps],m_next)*par.eps_w[eps]

            # Find Consumption for shock eps in the next period 
            c_next =  tools.interp_linear_1d(sol.m[i,t+1,1,t+1,par.Ba:,eps],sol.c[i,t+1,1,t+1,par.Ba:,eps],m_next)

            # Calculate the  marginal utility of consumption in the next period and add it to the weighted sum weighted by the wage shock
            E_margu_work += marg_u(c_next, par)*par.eps_w[eps]

        c = (par.beta*(1+par.r)*E_margu_work)**(1/-par.rho) # Backout consumption using the inverted Euler 
        m = par.a_grid + c # Calculate the endogenous grid 
        V = util(c,par) + par.beta*V_next_work # Calculate the deterministc value function

        # Save results
        sol.c[i,t,0,t,par.Ba:,0] = c
        sol.m[i,t,0,t,par.Ba:,0] = m 
        sol.V[i,t,0,t,par.Ba:,0] = V
        sol.EMU[i,t,0,t,par.Ba:,0] = E_margu_work

        # Handle the Budget constraint
        sol.c[i,t,0,t,:par.Ba,0] = np.linspace(1e-16,m[0]-1e-8,par.Ba)
        sol.m[i,t,0,t,:par.Ba,0] = np.linspace(1e-16,m[0]-1e-8,par.Ba)
        sol.V[i,t,0,t,:par.Ba,0] = value_of_choice_working(sol.m[i,t,0,t,:par.Ba,0] , sol.c[i,t,0,t,:par.Ba,0] , i, t, sol, par)

        
    else: # Can continue to study

        # Find expected value and expected marginal utility if working in next period  
        for eps in range(par.neps): #Calculates the expected value and expected marginal utility if working in the next period
            # Find pre decision income in the next periodg
            m_next = (1+par.r)*par.a_grid 
            
            # Find the value in the next period given m_next by interpolation and weight with the wage shock weight, and add it to the weighted sum
            V_next_work += tools.interp_linear_1d(sol.m[i,t+1,1,t+1,par.Ba:,eps],sol.V[i,t+1,1,t+1,par.Ba:,eps],m_next)*par.eps_w[eps]

            # Find Consumption for shock eps in the next period
            c_next =  tools.interp_linear_1d(sol.m[i,t+1,1,t+1,par.Ba:,eps],sol.c[i,t+1,1,t+1,par.Ba:,eps],m_next)

           # Calculate the  marginal utility of consumption in the next period and add it to the weighted sum weighted by the wage shock
            E_margu_work += marg_u(c_next, par)*par.eps_w[eps] 

        # Find value, consumption and marginal utility if studying in next period

        # Pre decision state conditional on studying 
        m_next = (1+par.r)*par.a_grid + transfer(i,par)

        # Find the value of continuing to study in the next period
        V_next_study = tools.interp_linear_1d(sol.m[i,t+1,0,t+1,:,0],sol.V[i,t+1,0,t+1,:,0],m_next)

        # Find consumption in the next period conditional on studying
        c_next_study = tools.interp_linear_1d(sol.m[i,t+1,0,t+1,:,0],sol.c[i,t+1,0,t+1,:,0],m_next) 

        # Find marginal utility of consumption conditional on styding in the next period
        margu_study = marg_u(c_next_study,par) 
    
        
        # Find Choice probabilities 
        V = np.array([V_next_study,V_next_work])

        V_max = V.max(axis=0) 

        log_sum = V_max + par.sigma_taste*np.log((np.sum(np.exp((V-V_max)/par.sigma_taste),axis=0)))

        ccp = np.exp((V-log_sum)/par.sigma_taste)
    

        # Expected marginal utility 
        E_margu = ccp[0]*margu_study+ccp[1]*E_margu_work
        
        # Find consumption by inverting the expected marginal utility 
        c_now = (par.beta*(1+par.r)*E_margu)**(1/-par.rho)
        
        # Calculate the endogenous grid 
        m_now = par.a_grid + c_now 

        # Find the value in period t - net of taste shocks and conditional on studying 
        V_now = util(c_now,par) + par.beta*log_sum
        

        # Use the Secoundary Upper Envelope to refine solution grids 
        # Sort the grids 
        m = sorted(m_now) 
        I = m_now 
        c = [x for _,x in sorted(zip(I,c_now))]
        V = [x for _,x in sorted(zip(I,V_now))]

        for  k in range(np.size(m_now)-2): # Go through "solution grid"
            # Fix values and calculate consumption slope
            m_low = m_now[k]
            m_high = m_now[k+1]
            c_slope = (c_now[k+1]-c_now[k])/(m_high-m_low)

            for j in range(len(m)): # Check values inside the sorted grid
                if m[j]>=m_low and m[j]<=m_high: 
                    c_geuss = c_now[k] + c_slope*(m[j]-m_low) # interpolate consumption 
                    V_geuss = value_of_choice_study(m[j], c_geuss, i, t, sol, par) # Calculate the value of choice

                    if V_geuss >V[j]: # Check what consumption dominates 
                        V[j] = V_geuss 
                        c[j] = c_geuss 
        
        # Return the solution
        sol.c[i,t,0,t,par.Ba:,0] = c
        sol.m[i,t,0,t,par.Ba:,0] = m 
        sol.V[i,t,0,t,par.Ba:,0] = V
        sol.ccp_work[i,t,0,t,par.Ba:,0] = ccp[1]
        sol.EMU[i,t,0,t,par.Ba:, 0] = E_margu

        # Handle the budget constraint
        sol.c[i,t,0,t,:par.Ba,0] = np.linspace(1e-16,m[0]-1e-8,par.Ba)
        sol.m[i,t,0,t,:par.Ba,0] = np.linspace(1e-16,m[0]-1e-8,par.Ba)
        sol.V[i,t,0,t,:par.Ba,0] = value_of_choice_study(sol.m[i,t,0,t,:par.Ba,0] ,sol.c[i,t,0,t,:par.Ba,0] , i, t, sol, par)
        sol.ccp_work[i,t,0,t,:par.Ba,0] = value_of_choice_study(sol.m[i,t,0,t,:par.Ba,0] ,sol.c[i,t,0,t,:par.Ba,0] , i, t, sol, par,probability= True)



def transfer(type,par): 
    '''
    Function that returns the transfers of a certain type 

    Parameters: 
        type, the type of the agent 
        par, namespace containing the parameters of the model

    Returns: 
        Transfer level
    '''
    if type == 0 or type == 2: 
        return par.phi_high 
    if type ==1 or type ==3: 
        return par.phi_low

def util(c,par): 
    ''' 
    Function that returns the flow utility of consumption 

    Parameters: 
        c, Consumption level 
        par, parameters of the model

    Returns: 
        Flow utility of consumption
    '''

    if par.rho == 1:
        return np.log(c)
    else:
        return c**(1-par.rho)/(1-par.rho)

def marg_u(c,par): 
    ''' 
    Function that returns the marginal utility of consumption

    Parameters: 
        c, Consumption level 
        par, parameters of the model

    Returns: 
        Marginal utility of consumption
    '''
    return c**(-par.rho)

def value_of_choice_study(m,c,type,t,sol,par, probability = False): 
    '''
    Calculates the (deterministic) value of choice conditional on studying 

    Paramters: 
        m, pre decision assets
        c, choosen level of consumption 
        type, index for type 
        t, index for time 
        sol, namespace for solution of the model
        par, namepace for parameters of the model
        probability, If true return choice probabilities only
    
    Returns: 
        Value of choice
    '''
    a = m-c # Calculate pose decision state 

    # Calculate the expected value conditional on choosing to work in the next period
    V_next_work = 0 

    for eps in range(par.neps):
        m_next = (1+par.r)*a
        try:
            V_next_work += tools.interp_linear_1d_scalar(sol.m[type,t+1,1,t+1,par.Ba:,eps],sol.V[type,t+1,1,t+1,par.Ba:,eps],m_next)*par.eps_w[eps]
        except: 
            V_next_work += tools.interp_linear_1d(sol.m[type,t+1,1,t+1,par.Ba:,eps],sol.V[type,t+1,1,t+1,par.Ba:,eps],m_next)*par.eps_w[eps]

    #2) Find value, consumption and marginal utility if studying in next period

    m_next = (1+par.r)*a + transfer(type,par)

    try:
        V_next_study = tools.interp_linear_1d_scalar(sol.m[type,t+1,0,t+1,:,0],sol.V[type,t+1,0,t+1,:,0],m_next)
    except: 
        V_next_study = tools.interp_linear_1d(sol.m[type,t+1,0,t+1,:,0],sol.V[type,t+1,0,t+1,:,0],m_next)
    
    # Calculate Choice probabilities 
    V = np.array([V_next_study,V_next_work])

    V_max = V.max(axis=0) 

    log_sum = V_max + par.sigma_taste*np.log((np.sum(np.exp((V-V_max)/par.sigma_taste),axis=0)))

    ccp = np.exp((V-log_sum)/par.sigma_taste)

    # Return 
    if probability: 
        return ccp[1] 

    return util(c,par) + par.beta*log_sum



def value_of_choice_working(m,c,type,t,sol,par):

    '''
    Calculates the (deterministic) value of choice conditional on working 

    Paramters: 
        m, pre decision assets
        c, choosen level of consumption 
        type, index for type 
        t, index for time 
        sol, namespace for solution of the model
        par, namepace for parameters of the model
    
    Returns: 
        Value of choice
    '''
    a = m - c # Calculate post decision state
    m_next = (1+par.r)*a # Find pre decision state in the next period
    V_next_work = 0

    # Interpolate the value in next period using m_next 

    for i_eps in range(par.neps): 
        try: 
            V_next_work += tools.interp_linear_1d_scalar(sol.m[type,t+1,1,t+1,par.Ba:,i_eps], sol.V[type,t+1,1,t+1,par.Ba:,i_eps], m_next)*par.eps_w[i_eps]
        except: 
            V_next_work += tools.interp_linear_1d(sol.m[type,t+1,1,t+1,par.Ba:,i_eps], sol.V[type,t+1,1,t+1,par.Ba:,i_eps], m_next)*par.eps_w[i_eps]

    # Return value of choice 
    return util(c,par) + par.beta*V_next_work

