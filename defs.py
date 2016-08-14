import math

# common constants
C =  2.99792458e+10
PI = 3.14159265358979
EM = 0.910938215e-27
E = -4.80320427e-10

# run setup
resolution_multiplier = 1
courant_factor = 3
COARSE_GRID_SIZE = 384 * resolution_multiplier
dx = 6.0 / COARSE_GRID_SIZE
wavelength = 8 * dx * resolution_multiplier
wavenumber = 2 * PI / wavelength
angular_frequency = 2 * PI * C / wavelength
periods = 4
pulse_duration = periods * wavelength / C
pulse_delay = 4 * pulse_duration
dt = dx / (C * courant_factor)
x0 = 0

PML_SIZE = 12

FINE_GRID_SIZE = 130 * resolution_multiplier # in coarse grid cell units
DEBUG_PADDING = 2
AUX_GRID_SIZE = 1

ITERATIONS = courant_factor * resolution_multiplier * 600
OUTPUT_PERIOD = courant_factor * resolution_multiplier * 12


# pulse description
def pulse_longitudinal_profile(t):
    return math.exp(-((t - pulse_delay)/pulse_duration)**2 * 2 * math.log(2))

def pulse(x, t):
    return (math.sin(wavenumber * x - angular_frequency * t) *
            pulse_longitudinal_profile(t))


# source fields
def left_b(x, t):
    return pulse(x, t)

def left_e(x, t):
    return pulse(x, t)

def right_b(x, t):
    return 0

def right_e(x, t):
    return 0

