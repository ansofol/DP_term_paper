# Import Packages 
import numpy as np 
from project_code.Simulation import simulate
from scipy import optimize as opt 
import copy as copy

# Define Esimation function for experiment I - Only p_high and p_low are estimated 
def estimate(obj,data,model,geuss,weighting_matrix="I"): 
    '''
    Function used to estimate p1 and p2 

    input: 
        obj, Criterion function 
        data, "true data"
        model, model object 
        geuss, initial geuss on paramter values 
        weighting_matrix, weighting matrix in the criteria function, default is the identity matrix 

    returns: 
        result, Estimates of p1,p2 and the criterion value at the estimates 
        xs, convergence path
    '''


    data_true = copy.deepcopy(data) # Save the "true data" - deepcopy required since we use sim as true data, which will be overwritten

    # Calculate the observed share of individuals who are rich and poor respectively 
    share_Rich = np.sum((data_true.type == 0) + (data_true.type == 2)) / len(data_true.type) 
    share_Poor = 1-share_Rich
    
    # Save shares
    list_shares = [share_Rich,share_Poor]

    # Define object to minimize, in this case the criterion function. 
    object = lambda x: obj(x,data_true,model=model,shares = list_shares, weighting_matrix=weighting_matrix)

    # callback func
    xs = []
    def callback(result):
        xs.append(result)

    result = opt.minimize(object, geuss, method = "Nelder-Mead", callback=callback)
    #result = opt.minimize(object, geuss, method = "Powell")

    return result, xs

def obj(x,data,model,shares, weighting_matrix): 
    '''
    Auxiliary Function used to evaluate the criterion function when estimating p_high and p_low 

    input: 
        x, parameter values
        data, true data 
        model, model object
        shares, shares of rich and poor individuals 
        weighting_matrix, weighting matrix used to construct the criterion function

    returns: 
        Criterion value
    '''
    
    # Construct distribution of types using the shares of rich and poor 
    p1 = shares[0]*x[0]
    p2 = shares[1]*x[1]
    p3 = shares[0] - p1
    p4 = shares[1] - p2
    p_list = [p1,p2,p3,p4] # Distribution 

    # Pass distribution of types to the criterion function alongside data, model and the weighting matrix 
    return  criteria(p_list, data, model, weighting_matrix)



def criteria(par_est,data,model,weighting_matrix="I"):

    '''
    Function that calculates the criterion value of given parameters and data for estimating p_high and p_low

    input: 
        par_est, parameters 
        data, true data 
        model, model object 
        weighting_matrix, weighting matrix to use when evaluating the criterion function, default is the identity matrix

    returns: 
        Criterion value
    '''

    moment_data = moments(data,model) # Calculate the moments used for the true data 

    setattr(model.par, "dist", par_est) # Changes the distribution in the model object
    
    # Unpack values and grids
    sol = model.sol
    sim = model.sim
    par = model.par 

    # Initialize 
    moment_sim = np.zeros(2*(model.par.Smax+2)) # Array to hold the simulation moments

    # Set seed to use in the estimation 
    seed_obj = np.random 
    seed_obj.seed(2023)
    par.random = seed_obj

    # Simulate N times and calculate the average moments over the N simulations 
    for i in range(par.Ns):
        reset_sim(sim, model)
        simulate(sim,sol,par)
        moment_sim +=  moments(sim,model)/model.par.Ns
   
    A = moment_data - moment_sim #Calculate the differences between the simulated moments and the data moments  

 
    # Return the criterion function
    if weighting_matrix == "I": 
        B  = np.eye(2*(par.Smax+2)) # Identity matrix 
        print(A @ B @ A.T)
        return A @ B @ A.T
        
    else: 
        B = weighting_matrix 
        return A @ B @ A.T 
        


def moments(data,model): 
    '''
    Function that calculates moments related to education 

    input: 
        data, data to calculate the moments over 
        model, model object 
    
    Returns: 
        A vector of 16 x 1 moments related to reducation 
    '''
    # Unpack parameters
    par = model.par

    
    moments = np.zeros(2*(par.Smax+2)) # Array to hold the moments 

    # calculate the share of rich and poor agents
    I_rich = (data.type == 0) + (data.type == 2)
    I_poor = (data.type == 1) + (data.type == 3)

    # Calculate moments 
    z = 0
    for k in [I_rich,I_poor]: # Loop over poor and rich
        for i in range(par.Smax+1): # loop over education length
            I_educ = np.max(data.S,axis=1) == i # Find people with i periods of education

            I = k*I_educ # Find interaction with income group

            moments[z+i] = np.sum(I)/np.sum(k)*100 # Find share of group k with i periods of education
        moments[z+par.Smax+1] = np.mean(data.S[k]) # add average periods of education for each group
        z += par.Smax+2 
    return moments # Return moments 


def reset_sim(sim,model): 
    '''
    Auxliary function that re-allocates the simulation grids 

    inputs: 
        sim, simple namespace that contains the simulation grids 
        model, model object 
    '''

    shape_sim = (model.par.N,model.par.Tsim)
    sim.c = np.zeros(shape_sim) 
    sim.S = np.zeros(shape_sim) 
    sim.ell = np.zeros(shape_sim) 
    sim.m = np.zeros(shape_sim) 
    sim.type = np.zeros(shape_sim) 
    sim.m[:,0] =  model.par.m_initial

    par = model.par
    par.phi = np.array([par.phi_high, par.phi_low,  par.phi_high, par.phi_low])

    

##  Extremely ambitious estimation thing :))))
def criterion_transfer(par_est, phi_high, data,model,weighting_matrix="I"):
    '''
    Function that calculates the criterion value of given parameters and data for estimating p_high, p_low and t(phi_high)

    input: 
        par_est, distribution parameters 
        phi_high, transfers received by agents of type phi_high, t(phi_high)
        data, true data 
        model, model object 
        weighting_matrix, weighting matrix to use when evaluating the criterion function, default is the identity matrix

    returns: 
        Criterion value
    '''

    moment_data = moments(data,model) # Calculate the empirical moments on the data used for estimation

    setattr(model.par, "dist", par_est) # Set the distribution parameters
    setattr(model.par, "phi_high", phi_high) # Set t(phi_high)
 
    model.solve_study() # Solve the study stage

    #Unpack values
    sol = model.sol
    sim = model.sim
    par = model.par 

    moment_sim = np.zeros(2*(model.par.Smax+2)) # Array that contains the simulated moments
    
    #Set seed
    seed_obj = np.random 
    seed_obj.seed(2023)
    par.random = seed_obj

    # Calculate the simulated moments of the N simulations and take the average over the N simulations
    for i in range(par.Ns):
        reset_sim(sim, model)
        simulate(sim,sol,par)
        moment_sim +=  moments(sim,model)/model.par.Ns
    A = moment_data - moment_sim # Calculate the differences between the data moments and the average simulated moments

    # calculate the criterion function and return the value
    if weighting_matrix == "I": 
        B  = np.eye(2*(par.Smax+2)) # Identity matrix 
        print(A @ B @ A.T)
        return A @ B @ A.T
        
    
    else: 
        B = weighting_matrix 
        return A @ B @ A.T 
    

def obj_transfer(x, data,model,shares, weighting_matrix):

    '''
    Auxiliary Function used to evaluate the criterion function when estimating p_high, p_low and t(phi_high) 

    input: 
        x, parameter values
        data, true data 
        model, model object
        shares, shares of rich and poor individuals 
        weighting_matrix, weighting matrix used to construct the criterion function

    returns: 
        Criterion value
    '''
     
    # Calculate the unconditional distribution given the parameters and the true shares of rich and poor agents
    p1 = shares[0]*x[0] 
    p2 = shares[1]*x[1]
    p3 = shares[0] - p1
    p4 = shares[1] - p2
    p_list = [p1,p2,p3,p4]

    phi_high = x[-1]

    # Pass the distribution alongside t(phi_high) to the criterion function and return the value
    return  criterion_transfer(p_list, phi_high, data, model, weighting_matrix)


def estimate_transfer(data,model,geuss,weighting_matrix="I"): 

    '''
    Function used to estimate p1, p2 and t(phi_high) 

    input: 
        obj, Criterion function 
        data, "true data"
        model, model object 
        geuss, initial geuss on paramter values 
        weighting_matrix, weighting matrix in the criteria function, default is the identity matrix 

    returns: 
        result, Estimates of p1, p2, t(phi_high) and the criterion value at the estimates 
    '''

    data_true = copy.deepcopy(data) # Save the "true data" - deepcopy required since we use sim as true data, which will be overwritten

    # Calculate observed income backgrounds 
    share_Rich = np.sum((data_true.type == 0) + (data_true.type == 2)) / len(data_true.type) 
    share_Poor = 1-share_Rich
    
    # Save shares 
    list_shares = [share_Rich,share_Poor]

    # Define object to minimize, in this case the criterion function
    object = lambda x: obj_transfer(x,data_true,model=model,shares = list_shares, weighting_matrix=weighting_matrix)

     
    bounds = ((0,1),(0,1),(0,100)) # Bounds passed to the optimizer

    result = opt.minimize(object, geuss, method = "Nelder-Mead",bounds=bounds) # Call optimizer 
    #result = opt.minimize(object, geuss, method = "Powell")

    return result # return estimation results 
