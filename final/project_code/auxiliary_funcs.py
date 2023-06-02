import numpy as np

def wage_func(i_S, t, i_type, eta, par):
        """
        Calculates the wage based on the given input parameters.

        Parameters:
        i_S (int): Education index.
        t (float): Time period.
        i_type (int):Type index.
        eta (float): Wage shock.
        par (object): An object containing necessary parameters for the calculation.

        Returns:
        float: The calculated wage based on the input parameters.

        """
        return np.exp(par.lambda_vec[i_S]*np.log(1+par.theta[i_type]) + eta)