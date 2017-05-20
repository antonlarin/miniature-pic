from math import log, exp

from defs import C

class Pml(object):
    """ Pml represents a collection of cells, comprising one
        PML layer. start and finish are sides of PML layer,
        which absorb the least and the most aggresively,
        respectively.
    """
    def __init__(self, start, finish, grid, r0=1e-8):
        self.start = start
        self.finish = finish
        self.size = abs(self.finish - self.start) + 1
        self.grid = grid

        self.n = 4
        self.max_sigma = -log(r0) * (self.n + 1) / (2 * self.size *
                grid.dx)
        self.b_coeffs, self.e_coeffs = self.compute_coeffs()

    def compute_coeffs(self):
        dx, dt = self.grid.dx, self.grid.dt
        coeffs = []
        left = min(self.start, self.finish)
        right = max(self.start, self.finish)
        for i in range(left, right + 1):
            # both b and e
            sigma = self.compute_sigma(i)
            if sigma < 1e-8:
                coeffs.append((1, 0.5 * C * dt / dx))
            else:
                exp_coeff = exp(-sigma * C * dt)
                coeffs.append((exp_coeff, 0.5 * (1 - exp_coeff) / sigma / dx))

        return (coeffs, coeffs)

    def compute_sigma(self, i):
        if self.start < self.finish:
            x = float(i - self.start) / (self.finish + 1 - self.start)
        else:
            x = float(self.start + 1 - i) / (self.start + 1 - self.finish)
        return self.max_sigma * x ** self.n

    def is_inside(self, i):
        if self.start < self.finish:
            return self.start <= i and i <= self.finish
        else:
            return self.finish <= i and i <= self.start

    def get_e_coeffs(self, i):
        left = min(self.start, self.finish)
        return self.e_coeffs[i - left]

    def get_b_coeffs(self, i):
        left = min(self.start, self.finish)
        return self.b_coeffs[i - left]
