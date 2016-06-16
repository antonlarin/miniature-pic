#!/usr/bin/env python

from __future__ import print_function
from sys import stdout, stderr, argv, exit
import numpy as np
import matplotlib.pyplot as plt

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

def update_b(grid, **kwargs):
    skip = kwargs.get('skip', lambda x: False)
    for i in range(1, grid.size):
        if not skip(i):
            A, B = grid.get_b_coeffs(i)
            grid.bs[i] = A * grid.bs[i] + B * (grid.es[i - 1] - grid.es[i])

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


# SG-DO operations
def correct_b_on_interface(grids, indices):
    b_source(grids['coarse'], indices['left_coarse_b'],
            grids['left_aux_coarse'].es[indices['left_aux_coarse_e']],
            'right')
    b_source(grids['coarse'], indices['right_coarse_b'],
            grids['right_aux_coarse'].es[indices['right_aux_coarse_e']],
            'left')

    b_source(grids['fine'], indices['left_fine_b'],
            grids['left_aux_fine'].es[indices['left_aux_fine_e']],
            'right')
    b_source(grids['fine'], indices['right_fine_b'],
            grids['right_aux_fine'].es[indices['right_aux_fine_e']],
            'left')

def correct_e_on_interface(grids, indices):
    e_source(grids['coarse'], indices['left_coarse_e'],
            grids['left_aux_coarse'].bs[indices['left_aux_coarse_b']],
            'right')
    e_source(grids['coarse'], indices['right_coarse_e'],
            grids['right_aux_coarse'].bs[indices['right_aux_coarse_b']],
            'left')

    e_source(grids['fine'], indices['left_fine_e'],
            grids['left_aux_fine'].bs[indices['left_aux_fine_b']],
            'right')
    e_source(grids['fine'], indices['right_fine_e'],
            grids['right_aux_fine'].bs[indices['right_aux_fine_b']],
            'left')

def transfer_b_to_aux_grids(grids, indices):
    grids['left_aux_coarse'].bs[indices['left_aux_coarse_b']] = (
            grids['fine'].bs[indices['left_fine_b']])
    grids['left_aux_fine'].bs[indices['left_aux_fine_b']] = (
            grids['coarse'].bs[indices['left_coarse_b']])

    grids['right_aux_coarse'].bs[indices['right_aux_coarse_b']] = (
            grids['fine'].bs[indices['right_fine_b']])
    grids['right_aux_fine'].bs[indices['right_aux_fine_b']] = (
            grids['coarse'].bs[indices['right_coarse_b']])


def build_plot(coarse_grid, fine_grid, idx):
    xs = [coarse_grid.x0 + coarse_grid.dx * i
            for i in range(len(coarse_grid.bs))]
    fine_xs = [fine_grid.x0 + fine_grid.dx * i
            for i in range(len(fine_grid.bs))]

    plt.clf()
    params = { 'basey': 10, 'nonposy': 'clip' }
    try:
        plt.semilogy(xs, coarse_grid.bs, 'r', label='CG', **params)
    except ValueError:
        pass

    try:
        plt.semilogy(fine_xs, fine_grid.bs, 'b', label='FG', **params)
    except ValueError:
        pass

    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('Bz')
    plt.ylim(1, 3e8)
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
    fine_grid_idx = (defs.COARSE_GRID_SIZE - defs.FINE_GRID_SIZE) / 2
    fine_x0 = (defs.x0 + fine_grid_idx * defs.dx -
            (defs.PML_SIZE + 1 + defs.DEBUG_PADDING) * defs.dx / ref_factor)
    fine_grid_size = (defs.FINE_GRID_SIZE * ref_factor +
        2 * (defs.PML_SIZE + 1 + defs.DEBUG_PADDING))

    fine_grid = Grid(fine_grid_size, fine_x0, defs.dx / ref_factor,
            defs.dt) # defs.dt / ref_factor)
    fine_grid.add_pml(Pml(defs.PML_SIZE, 1, fine_grid))
    fine_grid.add_pml(Pml(fine_grid_size - defs.PML_SIZE - 1,
            fine_grid_size - 2, fine_grid))

    # pmls inside coarse grid
    left_cip_start = fine_grid_idx + defs.DEBUG_PADDING 
    left_cip_finish = fine_grid_idx + defs.DEBUG_PADDING + defs.PML_SIZE - 1
    coarse_grid.add_pml(Pml(left_cip_start, left_cip_finish, coarse_grid))
    right_cip_start = (fine_grid_idx + defs.FINE_GRID_SIZE -
            defs.DEBUG_PADDING)
    right_cip_finish = (fine_grid_idx + defs.FINE_GRID_SIZE -
            defs.PML_SIZE + 1 - defs.DEBUG_PADDING)
    coarse_grid.add_pml(Pml(right_cip_start, right_cip_finish, coarse_grid))

    # aux grids
    aux_grid_size = (defs.AUX_GRID_SIZE + defs.DEBUG_PADDING +
            defs.PML_SIZE + 1)

    left_aux_coarse_x0 = (defs.x0 + fine_grid_idx * defs.dx -
            (aux_grid_size - defs.AUX_GRID_SIZE) * defs.dx)
    left_aux_coarse_grid = Grid(aux_grid_size, left_aux_coarse_x0,
            defs.dx, defs.dt)
    left_aux_coarse_grid.add_pml(Pml(defs.PML_SIZE, 1, left_aux_coarse_grid))

    left_aux_fine_x0 = (defs.x0 + fine_grid_idx * defs.dx -
            defs.dx / ref_factor)
    left_aux_fine_grid = Grid(aux_grid_size, left_aux_fine_x0,
            defs.dx / ref_factor, defs.dt) # defs.dt / ref_factor)
    left_aux_fine_grid.add_pml(Pml(defs.AUX_GRID_SIZE +
            defs.DEBUG_PADDING, aux_grid_size - 2, left_aux_fine_grid))

    right_aux_coarse_x0 = (defs.x0 +
            (fine_grid_idx + defs.FINE_GRID_SIZE - 1) * defs.dx)
    right_aux_coarse_grid = Grid(aux_grid_size, right_aux_coarse_x0,
            defs.dx, defs.dt)
    right_aux_coarse_grid.add_pml(Pml(defs.AUX_GRID_SIZE +
            defs.DEBUG_PADDING, aux_grid_size - 2, right_aux_coarse_grid))

    right_aux_fine_x0 = (defs.x0 + (fine_grid_idx + defs.FINE_GRID_SIZE) *
            defs.dx - (aux_grid_size - 1) * defs.dx / ref_factor)
    right_aux_fine_grid = Grid(aux_grid_size, right_aux_fine_x0,
            defs.dx / ref_factor, defs.dt) # defs.dt / ref_factor)
    right_aux_fine_grid.add_pml(Pml(defs.PML_SIZE, 1, right_aux_fine_grid))

    # pack grids and indices into dicts for ease of passing into functions
    grids = {
        'coarse': coarse_grid,
        'fine': fine_grid,
        'left_aux_coarse': left_aux_coarse_grid,
        'left_aux_fine': left_aux_fine_grid,
        'right_aux_coarse': right_aux_coarse_grid,
        'right_aux_fine': right_aux_fine_grid
    }
    indices = {
        'left_coarse_b': fine_grid_idx,
        'left_coarse_e': fine_grid_idx - 1,
        'left_fine_b': defs.PML_SIZE + defs.DEBUG_PADDING + 1,
        'left_fine_e': defs.PML_SIZE + defs.DEBUG_PADDING + 1,
        'left_aux_coarse_b': aux_grid_size - 1,
        'left_aux_coarse_e': aux_grid_size - 2,
        'left_aux_fine_b': 1,
        'left_aux_fine_e': 1,

        'right_coarse_b': fine_grid_idx + defs.FINE_GRID_SIZE,
        'right_coarse_e': fine_grid_idx + defs.FINE_GRID_SIZE,
        'right_fine_b': fine_grid_size - defs.PML_SIZE - defs.DEBUG_PADDING - 1,
        'right_fine_e': fine_grid_size - defs.PML_SIZE - defs.DEBUG_PADDING - 2,
        'right_aux_coarse_b': 1,
        'right_aux_coarse_e': 1,
        'right_aux_fine_b': aux_grid_size - 1,
        'right_aux_fine_e': aux_grid_size - 2
    }

    generate_b, generate_e = get_field_generator(coarse_grid)

    stdout.write('iteration ')
    for t in range(defs.ITERATIONS):
        t_str = '{}'.format(t)
        stdout.write(t_str)
        stdout.flush()

        cg_skip = lambda i: (
                i >= fine_grid_idx + defs.DEBUG_PADDING + defs.PML_SIZE and
                i < fine_grid_idx + defs.FINE_GRID_SIZE -
                    defs.DEBUG_PADDING - defs.PML_SIZE)

        # update b
        update_b(coarse_grid, skip=cg_skip)
        update_b(fine_grid)
        update_b(left_aux_coarse_grid)
        update_b(left_aux_fine_grid)
        update_b(right_aux_coarse_grid)
        update_b(right_aux_fine_grid)

        generate_b(t * coarse_grid.dt)
        correct_b_on_interface(grids, indices)
        transfer_b_to_aux_grids(grids, indices)

        # update e
        update_e(coarse_grid, skip=cg_skip)
        update_e(fine_grid)
        update_e(left_aux_coarse_grid)
        update_e(left_aux_fine_grid)
        update_e(right_aux_coarse_grid)
        update_e(right_aux_fine_grid)

        generate_e((t + 0.5) * coarse_grid.dt)
        correct_e_on_interface(grids, indices)

        if t % defs.OUTPUT_PERIOD == 0:
            build_plot(coarse_grid, fine_grid, t / defs.OUTPUT_PERIOD)

        stdout.write('\b' * len(t_str))

    stdout.write('\n')


if __name__ == '__main__':
    ref_factor = parse_args()
    simulate(ref_factor)

