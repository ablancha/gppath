import numpy as np
from scipy.stats import uniform
from gpsearch import Inputs


class AugmentedInputs(Inputs):
    """A class that augments an instance of `Inputs`
    with an extra (uniform) dimension for time.

    """

    def __init__(self, inputs, bounds=[0,1]):
        self.inputs = inputs
        self.domain = inputs.domain + [bounds]
        self.input_dim = inputs.input_dim + 1

    def set_bounds(self, bounds):
        self.domain[-1] = bounds

    def pdf(self, x):
        bd = self.domain[-1]
        return self.inputs.pdf(x[:,:-1]) / (bd[1]-bd[0])

    def pdf_jac(self, x):
        bd = self.domain[-1]
        pdf_jac_input = self.inputs.pdf_jac(x[:,:-1]) / (bd[1]-bd[0])
        pdf_jac = np.hstack((pdf_jac_input, np.zeros(x[:,-1].shape)))
        return pdf_jac

