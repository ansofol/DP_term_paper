import numpy as np 
from Simulation import simulate
from scipy import optimize as opt 
import copy as copy


def estimate(obj,data,model,geuss,weighting_matrix="I"): 

    data_true = copy.deepcopy(data)
    share_Rich = np.sum((data_true.type == 0) + (data_true.type == 2)) / len(data_true.type) 
    share_Poor = 1-share_Rich
    
    list_shares = [share_Rich,share_Poor]

    object = lambda x: obj(x,data_true,model=model,shares = list_shares, weighting_matrix=weighting_matrix)

    # callback func
    xs = []
    def callback(result):
        xs.append(result)

    result = opt.minimize(object, geuss, method = "Nelder-Mead", callback=callback)
    #result = opt.minimize(object, geuss, method = "Powell")

    return result, xs

def obj(x,data,model,shares, weighting_matrix): 
    
    p1 = shares[0]*x[0]
    p2 = shares[1]*x[1]
    p3 = shares[0] - p1
    p4 = shares[1] - p2
    p_list = [p1,p2,p3,p4]

    return  criteria(p_list, data, model, weighting_matrix)



def criteria(par_est,data,model,weighting_matrix="I"):

    moment_data = moments(data,model)

    setattr(model.par, "dist", par_est)
    #print(par_est)

    #model.set_grids()

    sol = model.sol
    sim = model.sim
    par = model.par 

    moment_sim = np.zeros(2*(model.par.Smax+2))
    seed_obj = np.random 
    seed_obj.seed(2023)
    par.random = seed_obj
    for i in range(par.Ns):
        reset_sim(sim, model)
        simulate(sim,sol,par)
        moment_sim +=  moments(sim,model)/model.par.Ns
    A = moment_data - moment_sim
    #print(f'Moment_sim is {np.round(moment_sim,4)}')
    #print(f'Moment_data is {np.round(moment_data,4)}')

 
    if weighting_matrix == "I": 
        B  = np.eye(2*(par.Smax+2))
        print(A @ B @ A.T)
        return A @ B @ A.T
        
    
    else: 
        B = weighting_matrix 
        return A @ B @ A.T 
        


def moments(data,model): 
    
    par = model.par
    moments = np.zeros(2*(par.Smax+2)) 

    I_rich = (data.type == 0) + (data.type == 2)
    I_poor = (data.type == 1) + (data.type == 3)

    z = 0
    for k in [I_rich,I_poor]:
        for i in range(par.Smax+1): 
            I_educ = np.max(data.S,axis=1) == i 

            I = k*I_educ

            moments[z+i] = np.sum(I)/np.sum(k)*100
        moments[z+par.Smax+1] = np.mean(data.S[k])
        z += par.Smax+2 
    return moments


def reset_sim(sim,model): 
        shape_sim = (model.par.N,model.par.Tsim)
        sim.c = np.zeros(shape_sim) 
        sim.S = np.zeros(shape_sim) 
        sim.ell = np.zeros(shape_sim) 
        sim.m = np.zeros(shape_sim) 
        sim.type = np.zeros(shape_sim) 
        sim.m[:,0] =  model.par.m_initial

    

##  Extremely ambitious estimation thing :))))
def criterion_transfer(par_est, phi_high, data,model,weighting_matrix="I"):

    moment_data = moments(data,model)

    setattr(model.par, "dist", par_est)
    setattr(model.par, "phi_high", phi_high)
    #print(par_est)

    #model.set_grids()
    model.solve_study()

    sol = model.sol
    sim = model.sim
    par = model.par 

    moment_sim = np.zeros(2*(model.par.Smax+2))
    
    seed_obj = np.random 
    seed_obj.seed(2023)
    par.random = seed_obj

    for i in range(par.Ns):
        reset_sim(sim, model)
        simulate(sim,sol,par)
        moment_sim +=  moments(sim,model)/model.par.Ns
    A = moment_data - moment_sim
    #print(f'Moment_sim is {np.round(moment_sim,4)}')
    #print(f'Moment_data is {np.round(moment_data,4)}')
    #print(par.phi_high)

 

    if weighting_matrix == "I": 
        B  = np.eye(2*(par.Smax+2))
        print(A @ B @ A.T)
        return A @ B @ A.T
        
    
    else: 
        B = weighting_matrix 
        return A @ B @ A.T 
    

def obj_transfer(x, data,model,shares, weighting_matrix): 
    
    p1 = shares[0]*x[0]
    p2 = shares[1]*x[1]
    p3 = shares[0] - p1
    p4 = shares[1] - p2
    p_list = [p1,p2,p3,p4]

    phi_high = x[-1]

    return  criterion_transfer(p_list, phi_high, data, model, weighting_matrix)


def estimate_transfer(data,model,geuss,weighting_matrix="I"): 

    data_true = copy.deepcopy(data)
    share_Rich = np.sum((data_true.type == 0) + (data_true.type == 2)) / len(data_true.type) 
    share_Poor = 1-share_Rich
    
    list_shares = [share_Rich,share_Poor]

    object = lambda x: obj_transfer(x,data_true,model=model,shares = list_shares, weighting_matrix=weighting_matrix)

    bounds = ((0,1),(0,1),(0,100))
    result = opt.minimize(object, geuss, method = "Nelder-Mead",bounds=bounds)
    #result = opt.minimize(object, geuss, method = "Powell")

    return result
