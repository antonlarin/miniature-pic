import numpy as np

from defs import C

class Grid(object):
    def __init__(self, size, dx, dt):
        self.size = size
        self.dx = dx
        self.dt = dt
        self.pmls = []
        self.es = np.zeros(size)
        self.bs = np.zeros(size)

    def add_pml(self, pml):
        self.pmls.append(pml)

    def e(self, i):
        return self.es[i]

    def b(self, i):
        return self.bs[i]

    def get_e_coeffs(self, i):
        for pml in self.pmls:
            if pml.is_inside(i):
                return pml.get_e_coeffs(i)

        return (1, C * self.dt / self.dx)

    def get_b_coeffs(self, i):
        for pml in self.pmls:
            if pml.is_inside(i):
                return pml.get_b_coeffs(i)

        return (1, C * self.dt / self.dx)

