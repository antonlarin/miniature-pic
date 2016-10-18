#!/usr/bin/env python

from __future__ import print_function, division
from sys import stdout, stderr, argv, exit
import numpy as np
import matplotlib.pyplot as plt
import math

import defs
from grid import Grid
from pml import Pml

# TFSF
def _source(grid, dst_field, dst_idx, src_value, direction):
    direction_factor = -1 if direction == 'left' else 1
    dst_field[dst_idx] += direction_factor * grid.cdt_by_dx * src_value

def b_source(grid, dst_idx, e_value, direction):
    _source(grid, grid.bs, dst_idx, e_value, direction)

def e_source(grid, dst_idx, b_value, direction):
    _source(grid, grid.es, dst_idx, b_value, direction)


# field update
def update_e(grid, **kwargs):
    skip = kwargs.get('skip', lambda x: False)
    for i in range(grid.size - 1):
        if not skip(i):
            A, B = grid.get_e_coeffs(i)
            grid.es[i] = A * grid.es[i] + B * (grid.bs[i] - grid.bs[i + 1])

def update_b(grid, first_half, **kwargs):
    skip = kwargs.get('skip', lambda x: False)
    for i in range(1, grid.size):
        if not skip(i):
            A, B = grid.get_b_coeffs(i)
            if first_half:
                if not grid.in_pml(i):
                    grid.bs[i] = (A * grid.bs[i] +
                            0.5 * B * (grid.es[i - 1] - grid.es[i]))
            else:
                if grid.in_pml(i):
                    grid.bs[i] = (A * grid.bs[i] +
                            B * (grid.es[i - 1] - grid.es[i]))
                else:
                    grid.bs[i] = (A * grid.bs[i] +
                            0.5 * B * (grid.es[i - 1] - grid.es[i]))


# field generation
def get_field_generator(coarse_grid):
    _fieldgen_idx = defs.PML_SIZE + 1;
    left_e_x = coarse_grid.x0 + coarse_grid.dx * (_fieldgen_idx + 0.5)
    right_e_x = coarse_grid.x0 + coarse_grid.dx * (defs.COARSE_GRID_SIZE -
            _fieldgen_idx - 0.5)
    def generate_b(t):
        # left
        b_source(coarse_grid, _fieldgen_idx, defs.left_e(left_e_x, t),
                'right')

        # right
        b_source(coarse_grid, defs.COARSE_GRID_SIZE - _fieldgen_idx,
                defs.right_e(right_e_x, t), 'left')

    left_b_x = coarse_grid.x0 + coarse_grid.dx * _fieldgen_idx
    right_b_x = coarse_grid.x0 + coarse_grid.dx * (defs.COARSE_GRID_SIZE -
            _fieldgen_idx)
    def generate_e(t):
        # left
        e_source(coarse_grid, _fieldgen_idx, defs.left_b(left_b_x, t),
                'right')

        # right
        e_source(coarse_grid, defs.COARSE_GRID_SIZE - _fieldgen_idx - 1,
                defs.right_b(right_b_x, t), 'left')

    return (generate_b, generate_e)


# transfers
def M(x):
    return math.sin(defs.PI * x) ** 2

def lagrange3_middle(f0, f1, f2, f3):
    return (-f0 + 9*f1 + 9*f2 - f3) / 16

def conduct_transfers(coarse_grid, fine_grid, transfer_params, t):
    left_cg_window_start = transfer_params['left_cg_window_start']
    left_cg_window_end = transfer_params['left_cg_window_end']
    left_fg_window_start = transfer_params['left_fg_window_start']
    left_fg_window_end = transfer_params['left_fg_window_end']
    ref_factor = transfer_params['ref_factor']

    # left interface
    # - compute F_m and substract them from the source grids
    # -- on coarse grid
    left_cg_window_x0 = coarse_grid.x0 + coarse_grid.dx * left_cg_window_start
    left_cg_window_x1 = coarse_grid.x0 + coarse_grid.dx * left_cg_window_end
    left_cg_window_dx = left_cg_window_x1 - left_cg_window_x0
    left_cg_window_indices = range(left_cg_window_start, left_cg_window_end)

    r2l_values = np.zeros(2 * defs.FFT_WINDOW_SIZE + 1)
    common_coeff = 0.5 * defs.TRANSFER_RATIO
    for i in left_cg_window_indices:
        local_i = i - left_cg_window_start
        mask = M(local_i / defs.FFT_WINDOW_SIZE)
        coeff = common_coeff * mask
        r2l_values[2 * local_i] = coeff * (coarse_grid.bs[i] +
                lagrange3_middle(coarse_grid.es[i - 2],
                    coarse_grid.es[i - 1],
                    coarse_grid.es[i],
                    coarse_grid.es[i + 1]))
        coarse_grid.bs[i] -= r2l_values[2 * local_i]

        mask = M((local_i + .5) / defs.FFT_WINDOW_SIZE)
        coeff = common_coeff * mask
        r2l_values[2 * local_i + 1] = coeff * (coarse_grid.es[i] +
                lagrange3_middle(coarse_grid.bs[i - 1],
                    coarse_grid.bs[i],
                    coarse_grid.bs[i + 1],
                    coarse_grid.bs[i + 2]))
        coarse_grid.es[i] -= r2l_values[2 * local_i + 1]

    mask = 0
    coeff = common_coeff * mask
    r2l_values[-1] = coeff * (coarse_grid.bs[left_cg_window_end] +
            lagrange3_middle(coarse_grid.es[left_cg_window_end - 2],
                coarse_grid.es[left_cg_window_end - 1],
                coarse_grid.es[left_cg_window_end],
                coarse_grid.es[left_cg_window_end + 1]))
    coarse_grid.bs[left_cg_window_end] -= r2l_values[-1]

    # cg_f_y = [coarse_grid.es[i] for i in left_cg_window_indices]
    # cg_f_y_xs = [coarse_grid.x0 + (i + 0.5) * coarse_grid.dx for i in
            # left_cg_window_indices]
# 
    # cg_f_z = [coarse_grid.bs[i]*1j for i in left_cg_window_indices]
    # cg_f_z_xs = [coarse_grid.x0 + i * coarse_grid.dx for i in
            # left_cg_window_indices]
# 
    # apply_M = lambda (x, v): v * M((x - left_cg_window_x0) / left_cg_window_dx)
    # cg_f_m_y = map(apply_M, zip(cg_f_y_xs, cg_f_y))
    # cg_f_m_z = map(apply_M, zip(cg_f_z_xs, cg_f_z))
# 
    # for i in left_cg_window_indices:
        # coarse_grid.es[i] -= cg_f_m_y[i - left_cg_window_start].real
        # coarse_grid.bs[i] -= cg_f_m_z[i - left_cg_window_start].imag

    # -- on fine grid
    left_fg_window_x0 = fine_grid.x0 + fine_grid.dx * left_fg_window_start
    left_fg_window_x1 = fine_grid.x0 + fine_grid.dx * left_fg_window_end
    left_fg_window_dx = left_fg_window_x1 - left_fg_window_x0
    left_fg_window_indices = range(left_fg_window_start, left_fg_window_end)

    for i in left_fg_window_indices:
        un_i = 2 * (i - left_fg_window_start)
        src_i = un_i // ref_factor
        right_coeff = (un_i - src_i * ref_factor) / ref_factor
        left_coeff = 1. - right_coeff
        fine_grid.bs[i] += (left_coeff * r2l_values[src_i] +
                right_coeff * r2l_values[src_i + 1])
        un_i += 1
        src_i = un_i // ref_factor
        right_coeff = (un_i - src_i * ref_factor) / ref_factor
        left_coeff = 1. - right_coeff
        fine_grid.es[i] += (left_coeff * r2l_values[src_i] +
                right_coeff * r2l_values[src_i + 1])
    fine_grid.bs[left_fg_window_end] += r2l_values[-1]

    # fg_f_y = [fine_grid.es[i] for i in left_fg_window_indices]
    # fg_f_y_xs = [fine_grid.x0 + (i + 0.5) * fine_grid.dx for i in
            # left_fg_window_indices]
# 
    # fg_f_z = [fine_grid.bs[i]*1j for i in left_fg_window_indices]
    # fg_f_z_xs = [fine_grid.x0 + i * fine_grid.dx for i in
            # left_fg_window_indices]
# 
    # apply_M = lambda (x, v): v * M((x - left_fg_window_x0) / left_fg_window_dx)
    # fg_f_m_y = map(apply_M, zip(fg_f_y_xs, fg_f_y))
    # fg_f_m_z = map(apply_M, zip(fg_f_z_xs, fg_f_z))

    # for i in left_fg_window_indices:
        # fine_grid.es[i] -= fg_f_m_y[i - left_fg_window_start].real
        # fine_grid.bs[i] -= fg_f_m_z[i - left_fg_window_start].imag

    # - compute Fourier transforms of F_m and filter them
    # fourier_cg_f_m_y = np.fft.fft(cg_f_m_y)
    # fourier_cg_f_m_z = np.fft.fft(cg_f_m_z)
    # fourier_fg_f_m_y = np.fft.fft(fg_f_m_y)
    # fourier_fg_f_m_z = np.fft.fft(fg_f_m_z)

    # if t % defs.OUTPUT_PERIOD == 0:
        # plt.clf()
        # freqs = np.fft.fftfreq(len(cg_f_m_y), coarse_grid.dx)
        # plt.scatter(freqs, map(abs, fourier_cg_f_m_y), label='$F=E_y$')
        # plt.scatter(freqs, map(abs, fourier_cg_f_m_z), 20, 'r', label='$F=iB_z$')
        # plt.vlines((-0.25/coarse_grid.dx, 0.25/coarse_grid.dx), 0, 10, 'g',
                # label=r'$\frac{k_x}{2\pi}=\frac{1}{4\Delta x}$')
        # plt.yscale('log')
        # plt.ylim(1e-10, 10)
        # plt.xlabel(r'$\frac{k_x}{2\pi}$')
        # plt.legend(loc='best')
        # plt.ylabel(r'$\left|\mathcal{F}(F)\right|$')
        # plt.savefig('fourier{0:06d}.png'.format(t // defs.OUTPUT_PERIOD))

    # -- values from coarse grid: only leave positive wavenumbers
    # fourier_cg_f_m_y[0], fourier_cg_f_m_z[0] = (0j, 0j)
    # for i in range((fourier_cg_f_m_y.size + 1) // 2, fourier_cg_f_m_y.size):
        # fourier_cg_f_m_y[i], fourier_cg_f_m_z[i] = (0j, 0j)

    # -- values from fine grid: only leave negative wavenumbers
    # for i in range((fourier_fg_f_m_y.size + 1) // 2):
        # fourier_fg_f_m_y[i], fourier_fg_f_m_z[i] = (0j, 0j)
    
    # - compute inverse transforms and add results to destination grids
    # -- inserting into fine grid
    # cg_f_m_y_prime = np.concatenate(([0j], np.fft.ifft(fourier_cg_f_m_y), [0j]))
    # cg_f_m_y_prime_xs = [
            # (coarse_grid.x0 +
                # (left_cg_window_start - 0.5 + i) * coarse_grid.dx)
            # for i in range(len(cg_f_m_y_prime))]
    # cg_f_m_z_prime = np.concatenate((np.fft.ifft(fourier_cg_f_m_z), [0j]))
    # cg_f_m_z_prime_xs = [
            # (coarse_grid.x0 +
                # (left_cg_window_start + i) * coarse_grid.dx)
            # for i in range(len(cg_f_m_z_prime))]

    # def interpolate(x, grid):
        # intervals = zip(grid[:-1], grid[1:])
        # for ((x1, v1), (x2, v2)) in intervals:
            # if x1 <= x and x < x2:
                # return (x - x1) * (v1 - v2) / (x1 - x2) + v1

    # for i in left_fg_window_indices:
        # fine_grid.es[i] += interpolate(
                # fine_grid.x0 + (i + 0.5) * fine_grid.dx,
                # zip(cg_f_m_y_prime_xs, cg_f_m_y_prime)
        # ).real
        # fine_grid.bs[i] += interpolate(
                # fine_grid.x0 + i * fine_grid.dx,
                # zip(cg_f_m_z_prime_xs, cg_f_m_z_prime)
        # ).imag

    # -- inserting into coarse grid
    # half_ref_factor = ref_factor // 2
    # fg_f_m_y_prime = np.fft.ifft(fourier_fg_f_m_y)
    # fg_f_m_z_prime = np.fft.ifft(fourier_fg_f_m_z)
    # for i in range(left_cg_window_end - left_cg_window_start):
        # coarse_grid.es[left_cg_window_start + i] += (
                # fg_f_m_y_prime[half_ref_factor + i * ref_factor].real
        # )
        # coarse_grid.bs[left_cg_window_start + i] += (
                # fg_f_m_z_prime[i * ref_factor].imag
        # )

    # right interface
    # not yet implemented


# utility
def build_plot(coarse_grid, fine_grid, idx):
    xs = [coarse_grid.x0 + coarse_grid.dx * i
            for i in range(len(coarse_grid.bs))]
    fine_xs = [fine_grid.x0 + fine_grid.dx * i
            for i in range(len(fine_grid.bs))]

    plt.clf()
    params = { 'basey': 10, 'nonposy': 'clip' }
    plt.subplot(211)
    try:
        plt.semilogy(xs, coarse_grid.bs, 'r', label='CG', **params)
    except ValueError:
        pass

    try:
        plt.semilogy(fine_xs, fine_grid.bs, 'b', label='FG', **params)
    except ValueError:
        pass

    plt.grid(which='major')
    plt.grid(which='minor')
    plt.legend(loc='best')
    plt.ylabel('Bz')
    plt.ylim(1e-6, 1)
    plt.xlim(xs[0], xs[-1])

    xs = map(lambda x: x + 0.5 * coarse_grid.dx, xs)
    fine_xs = map(lambda x: x + 0.5 * fine_grid.dx, fine_xs)
    plt.subplot(212)
    try:
        plt.semilogy(xs, coarse_grid.es, 'r', label='CG', **params)
    except ValueError:
        pass

    try:
        plt.semilogy(fine_xs, fine_grid.es, 'b', label='FG', **params)
    except ValueError:
        pass

    plt.grid(which='major')
    plt.grid(which='minor')
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('Ey')
    plt.ylim(1e-6, 1)
    plt.xlim(xs[0], xs[-1])

    plt.savefig('{0:06d}.png'.format(idx), dpi=120)

def parse_args():
    if len(argv) < 2:
        print('No factor passed\n\tUSAGE: ./hst.py ref_factor', file=stderr)
        exit(1)

    try:
        ref_factor = int(argv[1])
    except ValueError:
        print('Couldn\'t parse ref_factor', file=stderr)
        exit(1)
    return ref_factor

def simulate(ref_factor):
    # coarse grid
    coarse_grid = Grid(defs.COARSE_GRID_SIZE, defs.x0, defs.dx, defs.dt)
    coarse_grid.add_pml(Pml(defs.PML_SIZE, 1, coarse_grid))
    coarse_grid.add_pml(Pml(defs.COARSE_GRID_SIZE - defs.PML_SIZE - 1,
            defs.COARSE_GRID_SIZE - 2, coarse_grid))

    # fine grid
    fine_grid_idx = (defs.COARSE_GRID_SIZE - defs.FINE_GRID_SIZE) // 2
    fine_x0 = (defs.x0 + (fine_grid_idx - defs.FFT_WINDOW_SIZE) * defs.dx -
            (defs.PML_SIZE + 1) * defs.dx / ref_factor)
    fine_grid_size = (defs.FINE_GRID_SIZE * ref_factor +
        2 * (defs.PML_SIZE + 1 + defs.FFT_WINDOW_SIZE * ref_factor))

    fine_grid = Grid(fine_grid_size, fine_x0, defs.dx / ref_factor, defs.dt)

    # pack grids and indices into dicts for ease of passing into functions
    transfer_params = {
            'ref_factor': ref_factor,

            'left_cg_window_start': fine_grid_idx - defs.FFT_WINDOW_SIZE,
            'left_cg_window_end': fine_grid_idx,
            'left_fg_window_start': 1 + defs.PML_SIZE,
            'left_fg_window_end': (1 + defs.PML_SIZE +
                defs.FFT_WINDOW_SIZE * ref_factor)
    }

    generate_b, generate_e = get_field_generator(coarse_grid)

    stdout.write('iteration ')
    for t in range(defs.ITERATIONS):
        t_str = '{}'.format(t)
        stdout.write(t_str)
        stdout.flush()

        cg_skip = lambda i: (
                i >= fine_grid_idx + defs.PML_SIZE and
                i < fine_grid_idx + defs.FINE_GRID_SIZE -
                    defs.PML_SIZE)

        # update second half b
        update_b(coarse_grid, False) #, skip=cg_skip)
        update_b(fine_grid, False)
        generate_b(t * coarse_grid.dt)

        # update e
        update_e(coarse_grid) #, skip=cg_skip)
        update_e(fine_grid)
        generate_e((t + 0.5) * coarse_grid.dt)
        
        # update first half b
        update_b(coarse_grid, True) #, skip=cg_skip)
        update_b(fine_grid, True)

        conduct_transfers(coarse_grid, fine_grid, transfer_params, t)

        if t % defs.OUTPUT_PERIOD == 0:
            build_plot(coarse_grid, fine_grid, t // defs.OUTPUT_PERIOD)

        # if t == defs.ITERATIONS - 1:
            # pairs = [(defs.x0 + defs.dx * i, coarse_grid.bs[i]) for i in
                    # range(coarse_grid.size)]
            # pairs = filter(lambda (x, _): x > 0.4 and x < 0.8, pairs)
            # values = map(abs, zip(*pairs)[1])
            # print(max(values))

        stdout.write('\b' * len(t_str))

    stdout.write('\n')


if __name__ == '__main__':
    ref_factor = parse_args()
    simulate(ref_factor)

