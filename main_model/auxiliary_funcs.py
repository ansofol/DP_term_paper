import numpy as np

def wage_func(i_S, t, i_type, eta, par):
        return np.exp(par.lambda_vec[i_S]*np.log(1+par.theta[i_type]) + eta)