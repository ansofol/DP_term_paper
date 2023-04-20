import numpy as np
import sys
sys.path.append('../main_model')
from model import Model
import EGM

class WorkModel(Model):
    pass

    def solve(self):
        pass
        # last period

        # egm step

    def wage_func(self, i_S, t, theta, eta, par):
        return np.exp(par.lambda_vec[i_S]*np.log(theta) + eta)

