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
    # - compute alternative variables and substract them from source grid
    # -- on coarse grid
    left_cg_window_x0 = coarse_grid.x0 + coarse_grid.dx * left_cg_window_start
    left_cg_window_x1 = coarse_grid.x0 + coarse_grid.dx * left_cg_window_end
    left_cg_window_dx = left_cg_window_x1 - left_cg_window_x0
    left_cg_window_indices = range(left_cg_window_start, left_cg_window_end)

    l2r_values = np.zeros(2 * defs.FFT_WINDOW_SIZE + 1)
    common_coeff = 0.5 * defs.TRANSFER_RATIO
    for i in left_cg_window_indices:
        local_i = i - left_cg_window_start
        mask = M(local_i / defs.FFT_WINDOW_SIZE)
        coeff = common_coeff #* mask
        l2r_values[2 * local_i] = coeff * (coarse_grid.bs[i] +
                lagrange3_middle(coarse_grid.es[i - 2],
                    coarse_grid.es[i - 1],
                    coarse_grid.es[i],
                    coarse_grid.es[i + 1]))
        coarse_grid.bs[i] -= l2r_values[2 * local_i]

        mask = M((local_i + .5) / defs.FFT_WINDOW_SIZE)
        coeff = common_coeff #* mask
        l2r_values[2 * local_i + 1] = coeff * (coarse_grid.es[i] +
                lagrange3_middle(coarse_grid.bs[i - 1],
                    coarse_grid.bs[i],
                    coarse_grid.bs[i + 1],
                    coarse_grid.bs[i + 2]))
        coarse_grid.es[i] -= l2r_values[2 * local_i + 1]

    mask = 0
    coeff = common_coeff #* mask
    l2r_values[-1] = coeff * (coarse_grid.bs[left_cg_window_end] +
            lagrange3_middle(coarse_grid.es[left_cg_window_end - 2],
                coarse_grid.es[left_cg_window_end - 1],
                coarse_grid.es[left_cg_window_end],
                coarse_grid.es[left_cg_window_end + 1]))
    coarse_grid.bs[left_cg_window_end] -= l2r_values[-1]

    # -- on fine grid
    left_fg_window_x0 = fine_grid.x0 + fine_grid.dx * left_fg_window_start
    left_fg_window_x1 = fine_grid.x0 + fine_grid.dx * left_fg_window_end
    left_fg_window_dx = left_fg_window_x1 - left_fg_window_x0
    left_fg_window_indices = range(left_fg_window_start, left_fg_window_end)
    left_fg_window_sparse_indices = range(left_fg_window_start,
            left_fg_window_end, ref_factor)

    r2l_values = np.zeros(2 * defs.FFT_WINDOW_SIZE + 1)
    common_coeff = 0.5 * defs.TRANSFER_RATIO
    for i in left_fg_window_sparse_indices:
        local_i = i - left_fg_window_start
        local_sparse_i = local_i // ref_factor
        mask = M(local_i / (defs.FFT_WINDOW_SIZE * ref_factor))
        coeff = common_coeff #* mask
        r2l_values[2 * local_sparse_i] = coeff * (-fine_grid.bs[i] +
                lagrange3_middle(fine_grid.es[i - 2],
                    fine_grid.es[i - 1],
                    fine_grid.es[i],
                    fine_grid.es[i + 1]))
        fine_grid.bs[i] += r2l_values[2 * local_sparse_i]

        half_ref_factor = ref_factor // 2
        mask = M((local_i + half_ref_factor + .5) /
                (defs.FFT_WINDOW_SIZE * ref_factor))
        coeff = common_coeff #* mask
        r2l_values[2 * local_sparse_i + 1] = coeff * (
                fine_grid.es[i + half_ref_factor] -
                lagrange3_middle(fine_grid.bs[i + half_ref_factor - 1],
                    fine_grid.bs[i + half_ref_factor],
                    fine_grid.bs[i + half_ref_factor + 1],
                    fine_grid.bs[i + half_ref_factor + 2]))
        fine_grid.es[i + half_ref_factor] -= r2l_values[2 * local_sparse_i + 1]

    mask = 0
    coeff = common_coeff #* mask
    r2l_values[-1] = coeff * (-fine_grid.bs[left_fg_window_end] +
            lagrange3_middle(fine_grid.es[left_fg_window_end - 2],
                fine_grid.es[left_fg_window_end - 1],
                fine_grid.es[left_fg_window_end],
                fine_grid.es[left_fg_window_end + 1]))
    fine_grid.bs[left_fg_window_end] += r2l_values[-1]

    # - insert alternative variables into target grids
    # -- in fine grid
    for i in left_fg_window_indices:
        un_i = 2 * (i - left_fg_window_start)
        src_i = un_i // ref_factor
        right_coeff = (un_i - src_i * ref_factor) / ref_factor
        left_coeff = 1. - right_coeff
        fine_grid.bs[i] += (left_coeff * l2r_values[src_i] +
                right_coeff * l2r_values[src_i + 1])
        un_i += 1
        src_i = un_i // ref_factor
        right_coeff = (un_i - src_i * ref_factor) / ref_factor
        left_coeff = 1. - right_coeff
        fine_grid.es[i] += (left_coeff * l2r_values[src_i] +
                right_coeff * l2r_values[src_i + 1])

    fine_grid.bs[left_fg_window_end] += l2r_values[-1]

    # -- in coarse grid
    fourier_r2l_values = np.fft.fft(r2l_values)
    freqs = np.fft.fftfreq(len(r2l_values), 0.5 * coarse_grid.dx)

    lowpass = lambda (f, v): 0j if abs(f) > (0.25 / coarse_grid.dx) else v
    filtered_r2l_values = map(lowpass, zip(freqs, fourier_r2l_values))
    r2l_values_prime = map(lambda z: z.real, np.fft.ifft(filtered_r2l_values))

    # if t % defs.OUTPUT_PERIOD == 0:
        # plt.clf()
        # plt.scatter(freqs, map(abs, fourier_r2l_values), 20, 'b',
                # label=r'$\frac{E_y - B_z}{2}$')
        # plt.scatter(freqs, map(abs, filtered_r2l_values), 20, 'r',
                # label=r'$H\left(\frac{E_y - B_z}{2}\right)$')
        # plt.vlines((-0.25/coarse_grid.dx, 0.25/coarse_grid.dx), 0, 10, 'g',
                # label=r'$\frac{k_x}{2\pi}=\frac{1}{4\Delta x}$')
        # plt.yscale('log')
        # plt.ylim(1e-10, 10)
        # plt.xlabel(r'$\frac{k_x}{2\pi}$')
        # plt.ylabel(r'$\left|\mathcal{F}(F)\right|$')
        # plt.legend(loc='best')
        # plt.savefig('fourier{0:06d}.png'.format(t // defs.OUTPUT_PERIOD))

    for i in left_cg_window_indices:
        un_i = 2 * (i - left_cg_window_start)
        coarse_grid.bs[i] -= r2l_values_prime[un_i]

        coarse_grid.es[i] += r2l_values_prime[un_i + 1]

    coarse_grid.bs[left_cg_window_end] -= r2l_values_prime[-1]

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
    fine_grid_start = (defs.COARSE_GRID_SIZE - defs.FINE_GRID_SIZE) // 2
    fine_grid_end = fine_grid_start + defs.FINE_GRID_SIZE
    fine_x0 = (defs.x0 + (fine_grid_start - defs.FFT_WINDOW_SIZE) * defs.dx -
            defs.dx / ref_factor)
    fine_fft_window_size = defs.FFT_WINDOW_SIZE * ref_factor
    fine_grid_size = (defs.FINE_GRID_SIZE * ref_factor +
        2 * (1 + fine_fft_window_size))

    fine_grid = Grid(fine_grid_size, fine_x0, defs.dx / ref_factor, defs.dt)

    # pack grids and indices into dicts for ease of passing into functions
    transfer_params = {
            'ref_factor': ref_factor,

            'left_cg_window_start': fine_grid_start - defs.FFT_WINDOW_SIZE,
            'left_cg_window_end': fine_grid_start,
            'left_fg_window_start': 1,
            'left_fg_window_end': 1 + defs.FFT_WINDOW_SIZE * ref_factor,

            'right_cg_window_start': fine_grid_end,
            'right_cg_window_end': fine_grid_end + defs.FFT_WINDOW_SIZE,
            'right_fg_window_start': fine_grid_size - 1 - fine_fft_window_size,
            'right_fg_window_end': fine_grid_size - 1
    }

    generate_b, generate_e = get_field_generator(coarse_grid)

    stdout.write('iteration ')
    for t in range(defs.ITERATIONS):
        t_str = '{}'.format(t)
        stdout.write(t_str)
        stdout.flush()

        cg_skip = lambda i: (
                i >= fine_grid_start + defs.PML_SIZE and
                i < fine_grid_end - defs.PML_SIZE)

        # update second half b
        update_b(coarse_grid, False, skip=cg_skip)
        update_b(fine_grid, False)
        generate_b(t * coarse_grid.dt)

        # update e
        update_e(coarse_grid, skip=cg_skip)
        update_e(fine_grid)
        generate_e((t + 0.5) * coarse_grid.dt)
        
        # update first half b
        update_b(coarse_grid, True, skip=cg_skip)
        update_b(fine_grid, True)

        conduct_transfers(coarse_grid, fine_grid, transfer_params, t)

        if t % defs.OUTPUT_PERIOD == 0:
            build_plot(coarse_grid, fine_grid, t // defs.OUTPUT_PERIOD)

        stdout.write('\b' * len(t_str))

    stdout.write('\n')


if __name__ == '__main__':
    ref_factor = parse_args()
    simulate(ref_factor)

