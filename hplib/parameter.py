import numpy as np
import math

class Parameter:
    """
    Parameter Base class
    """
    def __init__(self, initialization):
        pass

    def sample(self):
        pass

    def tostring(self):
        pass

class ParameterDiscrete(Parameter):
    """
    Parameter class for discrete elements of parameter space
    """
    def __init__(self, values):
        self.values = values
        
    def sample(self):
        return np.random.choice(self.values)

    def tostring(self):
        return "discrete " + str(self.values)

class ParameterUniform(Parameter):
    """
    Parameter class for uniformly distributed parameter space
    """
    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high
        
    def sample(self):
        return np.random.uniform(self.low, self.high)

    def tostring(self):
        return "uniform (" + str(self.low) + ", " + str(self.high) + ")"

class ParameterLogUniform(Parameter):
    """
    Parameter class for log uniformly distributed parameter space
    Base = 10
    E.g. learning rate, regularization
    """
    def __init__(self, low=10e-6, high=10e-1):
        self.loglow = math.log10(low)
        self.loghigh = math.log10(high)
        
    def sample(self):
        return (10**np.random.uniform(self.loglow, self.loghigh))

    def tostring(self):
        return "log uniform 10**(" + str(self.loglow) + ", " + str(self.loghigh) + ")"
