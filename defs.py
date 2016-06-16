import math

# common constants
COARSE_GRID_SIZE = 1024
PML_SIZE = 12
C =  2.99792458e+10
PI = 3.14159265358979
EM = 0.910938215e-27
E = -4.80320427e-10

wavelength = 0.8e-4
W0 = 2*PI*C/wavelength
REL_FIELD = -2*PI*EM*C*C/(E*wavelength)
dx = wavelength / 32
dt = dx/(4*C)
x0 = -COARSE_GRID_SIZE * dx / 2

FINE_GRID_SIZE = 256 # in coarse grid cell units
DEBUG_PADDING = 2
AUX_GRID_SIZE = 1

ITERATIONS = 5000
OUTPUT_PERIOD = 200


# pulse description
def sgn(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

def block(x, xmin, xmax):
    return (sgn(x - xmin) + sgn(xmax - x)) * 0.5

def f(t):
    return math.sin(W0*t/20)

def form(t):
    return f(t)*f(t)*f(t*20)*block(t, 0, 20*PI/W0)

DELAY = 16 * wavelength / C

def shape(r, t):
    return 2 * REL_FIELD * form(t + r/C - DELAY)

def pulse(x, t):
    return shape(abs(x), t)


# source fields
def left_b(x, t):
    return pulse(x, t)

def left_e(x, t):
    return pulse(x, t)

def right_b(x, t):
    return -pulse(x, t)

def right_e(x, t):
    return pulse(x, t)

