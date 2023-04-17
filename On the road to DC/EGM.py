import numpy as np 
import tools as tools


def EGM(s,d,educ,sol,par): 

    m =  np.zeros(par.Na)
    c =  np.zeros(par.Na)
    v =  np.zeros(par.Na)

    # raw EGM 
    if d == 1: # Working
        m_next = par.R*par.grid_a[s,:] + par.income_work[educ] # Next period cash on hand  
        
        v_next = tools.interp_linear_1d(sol.m[s+1,d,educ,par.N_bottom:],sol.v[s+1,d,educ,par.N_bottom:], m_next) 
        c_next = tools.interp_linear_1d(sol.m[s+1,d,educ,par.N_bottom:],sol.c[s+1,d,educ,par.N_bottom:], m_next)

        E_marg_util = marg_util(c_next, par)

        c = par.R*par.beta*(1/E_marg_util) # Using inverse expected marginal utility to find consumption in current period 

        m = c + par.grid_a[s,:] 

        v = util(c,par) + par.beta*v_next

        return m,c,v
        
    else:  # Studying
        
        # next period states 
        m_next_study = par.R*par.grid_a[s,:] + par.SU
        m_next_work = par.R*par.grid_a[s,:] + par.income_work[educ] 

        v_next = np.nan + np.zeros((2,par.Na))
        c_next = np.nan + np.zeros((2,par.Na))

        # Value studying next period
        v_next[0,:] = tools.interp_linear_1d(sol.m[s+1,0,educ+1,par.N_bottom:],sol.v[s+1,0,educ+1,par.N_bottom:], m_next_study) 
        c_next[0,:] = tools.interp_linear_1d(sol.m[s+1,0,educ+1,par.N_bottom:],sol.c[s+1,0,educ+1,par.N_bottom:], m_next_study) 

        # Value Working next period
        v_next[1,:] = tools.interp_linear_1d(sol.m[s+1,1,educ+1,par.N_bottom:],sol.v[s+1,1,educ+1,par.N_bottom:], m_next_work) 
        c_next[1,:] = tools.interp_linear_1d(sol.m[s+1,1,educ+1,par.N_bottom:],sol.c[s+1,1,educ+1,par.N_bottom:], m_next_work) 
        # 1 calculate logsum (here only works for sigma > 0) (for now) 
        v_max = v_next.max(axis=0)
        log_sum = v_max + par.sigma*(np.log(np.sum(np.exp((v_next-v_max)/par.sigma),axis=0)))
        
        #print(v_next.shape)
        # 2 calculate probabilities 
        prob = np.exp((v_next-log_sum)/par.sigma)
        print(np.round(prob[0],2))


        # 3 Calculate expected marginal utilities 
        marg_u_next = marg_util(c_next,par) 

        avg_marg_u_next = prob[0,:]* marg_u_next[0] + prob[1,:]* marg_u_next[1] 

        
        c = 1/avg_marg_u_next
        v = util(c,par) + par.beta*log_sum
        m = c + par.grid_a[s,:]

        m_raw = m 
        c_raw = c
        v_raw = v
        # Upper envelope :( 
        m = sorted(m_raw)
        I = m_raw
        c = [x for _,x in sorted(zip(I,c_raw))] 
        v = [x for _,x in sorted(zip(I,v_raw))]
        
        for i in range(np.size(m_raw)-2): 
            m_low = m_raw[i]
            m_high = m_raw[i+1] 
            c_slope = ((c_raw[i+1]-c_raw[i]))/(m_high-m_low)

            for j in range (len(m)): 
                if m[j] > m_low and m[j] <= m_high: 
                    c_geuss = c_raw[i]+c_slope*(m[j]-m_low)
                    v_geuss = value_of_choice_study(m[j],c_geuss,s,educ,sol,par)

                    if v_geuss >= v[j]: 
                        v[j] = v_geuss
                        c[j] = c_geuss

        
        return m,c,v

# Solve conditional on choice 

# Get raw values 

# Find the secoundary upper envelope (conditional on status - if working not needed)


#def studying 

#def working 




def value_of_choice_study(m,c,s,educ,sol,par): 
    a = m-c
    
    if a.size == 1: 
        m_next_study = np.array([par.R*a + par.SU]) 
        m_next_work = np.array([par.R*a + par.income_work[educ]]) 

    else: 
        m_next_study = par.R*a + par.SU
        m_next_work = par.R*a + par.income_work[educ]

    
     # Value studying next period
    v_next_0 = tools.interp_linear_1d(sol.m[s+1,0,educ+1,par.N_bottom:],sol.v[s+1,0,educ+1,par.N_bottom:],m_next_study)
    v_next_1 = tools.interp_linear_1d(sol.m[s+1,1,educ+1,par.N_bottom:],sol.v[s+1,1,educ+1,par.N_bottom:],m_next_work) 

    # 1 calculate logsum (here only works for sigma > 0) (for now) 
    v_next = np.array([v_next_0,v_next_1])
    v_max = v_next.max()
    log_sum = v_max + par.sigma*(np.log(np.sum(np.exp(v_next-v_max)/par.sigma,axis=0)))

    v = util(c,par) + par.beta*log_sum
    return v

def util(c,par): 
    return np.log(c) 

def marg_util(c,par): 
    return 1/c 

#def logsum 