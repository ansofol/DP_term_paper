import numpy as np 
from Simulation import simulate
from scipy import optimize as opt 


def estimate(data,model,weighting_matrix="I"): 

    object = lambda x: obj(x,data,model=model,weighting_matrix=weighting_matrix)

    result = opt.minimize(object, model.par.dist, method = "Nelder-Mead")

    return result


def obj(x,data,model,weighting_matrix): 
    p1 = x[0]
    p2 = x[1]
    p3 = x[2]
    p4 = 1 - p1 - p2 -p3 

    p_list = [p1,p2,p3,p4]

    return  criteria(p_list, data, model, weighting_matrix)



def criteria(par_est,data,model,weighting_matrix="I"):

    moment_data = moments(data,model)

    setattr(model.par, "dist", par_est)
    print(par_est)

    model.set_grids()

    sol = model.sol
    sim = model.sim
    par = model.par 

    moment_sim = np.zeros(2*(model.par.Smax+1))
    par.random.seed(2023)
    for i in range(par.Ns):
        reset_sim(sim, model)
        simulate(sim,sol,par)
        moment_sim +=  moments(sim,model)/model.par.Ns
    A = moment_data - moment_sim
    print(f'Moment_sim is {np.round(moment_sim,4)}')
    print(f'Moment_data is {np.round(moment_data,4)}')

 
    if weighting_matrix == "I": 
        B  = np.eye(2*(par.Smax+1))
        #print(A @ B @ A.T)
        return A @ B @ A.T
        
    
    else: 
        B = weighting_matrix 
        return A @ B @ A.T 
        


def moments(data,model): 
    
    par = model.par
    moments = np.zeros(2*(par.Smax+1)) 

    I_rich = (data.type == 0) + (data.type == 2)
    I_poor = (data.type == 1) + (data.type == 3)

    z = 0
    for k in [I_rich,I_poor]:
        for i in range(par.Smax+1): 
            I_educ = np.max(np.max(data.S,axis=1)) == i 

            I = k*I_educ

            moments[z+i] = np.sum(I)/np.sum(k)
        z += par.Smax+1 
    return moments 


def reset_sim(sim,model): 
        shape_sim = (model.par.N,model.par.Tsim)
        sim.c = np.zeros(shape_sim) 
        sim.S = np.zeros(shape_sim) 
        sim.ell = np.zeros(shape_sim) 
        sim.m = np.zeros(shape_sim) 
        sim.type = np.zeros(shape_sim) 
        sim.m[:,0] =  model.par.m_initial

    


    