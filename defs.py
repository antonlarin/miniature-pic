import math

import numpy as np

# common constants
C =  2.99792458e+10
PI = 3.14159265358979
EM = 0.910938215e-27
E = -4.80320427e-10

# run setup
resolution_multiplier = 3
courant_factor = 3
wavelength = 0.125 # cm
cells_per_lambda = 8
dx = wavelength / (cells_per_lambda * resolution_multiplier)
COARSE_GRID_SIZE = int(70 * wavelength / dx)
wavenumber = 2 * PI / wavelength
angular_frequency = 2 * PI * C / wavelength
pulse_width = 20 * wavelength
dt = dx / (C * courant_factor)
x0 = 0

PML_SIZE = 12
FINE_GRID_SIZE = int(22 * wavelength / dx)


ITERATIONS = int(90 * wavelength / C / dt)
OUTPUT_PERIOD = int(courant_factor * resolution_multiplier * 12)
ENERGY_OUTPUT_PERIOD = 1 # courant_factor * resolution_multiplier * 8


FFT_WINDOW_SIZE = 2 * cells_per_lambda * resolution_multiplier
TRANSFER_RATIO = 0.4
def transfer_mask(x):
    return math.sin(PI * x) ** 2

# pulse description
def pulse_longitudinal_profile(x, t):
    def block(x, xmin, xmax):
        return 1 if xmin <= x and x <= xmax else 0

    ref_x = (x - C * t) / pulse_width + 1.0
    if ref_x < 0.3:
        envelope_value = math.sin(ref_x / 0.6 * math.pi)**2
    elif ref_x < 0.7:
        envelope_value = 1
    else: # ref_x >= 0.7
        envelope_value = math.sin((ref_x - 0.4) / 0.6 * math.pi)**2

    return block(ref_x, 0, 1) * envelope_value

def pulse(x, t):
    return (math.sin(wavenumber * x - angular_frequency * t) *
            pulse_longitudinal_profile(x, t))


# source fields
def left_b(x, t):
    return pulse(x, t)
    # return 0

def left_e(x, t):
    return pulse(x, t)
    # return 0

def right_b(x, t):
    return 0
    # return -pulse(-x, t)

def right_e(x, t):
    return 0
    # return pulse(-x, t)
