#!/usr/bin/env python

from __future__ import print_function, division
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse

import defs
from grid import Grid
from pml import Pml
from transfer import Transfer


# TFSF
def _source(grid, dst_field, dst_idx, src_value, direction):
    direction_factor = -1 if direction == 'left' else 1
    dst_field[dst_idx] += direction_factor * 0.5 * grid.cdt_by_dx * src_value

def b_source(grid, dst_idx, e_value, direction):
    _source(grid, grid.bs, dst_idx, e_value, direction)

def e_source(grid, dst_idx, b_value, direction):
    _source(grid, grid.es, dst_idx, b_value, direction)


# field generation
def get_field_generator(coarse_grid):
    left_tfsf_plane_idx = defs.PML_SIZE + 2;
    right_tfsf_plane_idx = defs.COARSE_GRID_SIZE - defs.PML_SIZE - 2;

    target_indices = [
        left_tfsf_plane_idx - 1,
        left_tfsf_plane_idx,
        right_tfsf_plane_idx,
        right_tfsf_plane_idx + 1]
    source_xs = list(map(coarse_grid.x_of, [
        left_tfsf_plane_idx,
        left_tfsf_plane_idx - 1,
        right_tfsf_plane_idx + 1,
        right_tfsf_plane_idx]))
    directions = ['right'] * 2 + ['left'] * 2
    e_source_functions = [defs.left_e] * 2 + [defs.right_e] * 2
    b_source_functions = [defs.left_b] * 2 + [defs.right_b] * 2

    def add_source_currents_to_b(t):
        for idx, x, s, d in zip(target_indices,
                                source_xs,
                                e_source_functions,
                                directions):
            b_source(coarse_grid, idx, s(x, t), d)

    def add_source_currents_to_e(t):
        for idx, x, s, d in zip(target_indices,
                                source_xs,
                                b_source_functions,
                                directions):
            e_source(coarse_grid, idx, s(x, t), d)

    return (add_source_currents_to_b, add_source_currents_to_e)


# field update
def update_e(grid, **kwargs):
    skip = kwargs.get('skip', lambda x: False)
    for i in range(1, grid.size - 2):
        if not skip(i):
            A, B = grid.get_e_coeffs(i)
            grid.es[i] = A * grid.es[i] + B * (grid.bs[i - 1] - grid.bs[i + 1])

def update_half_b(grid, **kwargs):
    skip = kwargs.get('skip', lambda x: False)
    first_half = kwargs.get('first_half', False)
    second_half = not first_half

    for i in range(2, grid.size - 1):
        if not skip(i) and (second_half or not grid.in_pml(i)):
            A, B = grid.get_b_coeffs(i)
            grid.bs[i] = A * grid.bs[i] + B * (grid.es[i - 1] - grid.es[i + 1])


# utility
def build_plot(coarse_grid, fine_grid, pic_idx, output_dir):
    xs = [coarse_grid.x_of(i) for i in range(len(coarse_grid.bs))]
    fine_xs = [fine_grid.x_of(i) for i in range(len(fine_grid.bs))]

    plt.clf()
    plt.gcf().set_size_inches(8, 10)
    params = { 'basey': 10, 'nonposy': 'clip' }
    plt.subplot(211)
    try:
        plt.semilogy(xs,np.absolute(coarse_grid.bs), 'r', label='CG', **params)
    except ValueError:
        pass

    try:
        plt.semilogy(fine_xs, np.absolute(fine_grid.bs), 'b', label='FG', **params)
    except ValueError:
        pass

    plt.grid(which='major')
    plt.grid(which='minor')
    plt.legend(loc='best')
    plt.ylabel('Bz')
    plt.ylim(1e-8, 1)
    plt.xlim(xs[0], xs[-1])

    plt.subplot(212)
    try:
        plt.semilogy(xs, np.absolute(coarse_grid.es), 'r', label='CG', **params)
    except ValueError:
        pass

    try:
        plt.semilogy(fine_xs, np.absolute(fine_grid.es), 'b', label='FG', **params)
    except ValueError:
        pass

    plt.grid(which='major')
    plt.grid(which='minor')
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('Ey')
    plt.ylim(1e-8, 1)
    plt.xlim(xs[0], xs[-1])

    filename_pattern = output_dir + os.sep + '{0:06d}.png'
    plt.savefig(filename_pattern.format(pic_idx), dpi=120)


def calculate_energy(coarse_grid, fine_grid, ref_factor, transfer_params):
    coarse_grid_energy = 0
    fine_grid_energy = 0
    left_transfer_region_energy = 0
    right_transfer_region_energy = 0

    cg_indices = (
        list(range(defs.PML_SIZE + 1,
                   transfer_params['left_cg_window_start'])) +
        list(range(transfer_params['left_cg_window_end'],
                   transfer_params['left_cg_window_end'] + 2)) +
        list(range(transfer_params['right_cg_window_start'] - 2,
                   transfer_params['right_cg_window_start'])) +
        list(range(transfer_params['right_cg_window_end'],
                   coarse_grid.size - defs.PML_SIZE - 1)))

    for i in cg_indices:
        e = coarse_grid.es[i]
        b = coarse_grid.bs[i]
        coarse_grid_energy += (b * b + e * e) * coarse_grid.dx

    cg_left_transfer_region_indices = range(
            transfer_params['left_cg_window_start'],
            transfer_params['left_cg_window_end'])
    fg_left_transfer_region_indices = range(
            transfer_params['left_fg_window_start'],
            transfer_params['left_fg_window_end'], ref_factor)
    left_transfer_region_indices = zip(cg_left_transfer_region_indices,
            fg_left_transfer_region_indices)
    for cg_i, fg_i in left_transfer_region_indices:
        cg_e = coarse_grid.es[cg_i]
        cg_b = coarse_grid.bs[cg_i]

        for j in range(ref_factor):
            fg_e = fine_grid.es[fg_i + j]
            fg_b = fine_grid.bs[fg_i + j]

            left_transfer_region_energy += (
                    ((cg_b + fg_b)**2 + (cg_e + fg_e)**2) * fine_grid.dx)

    cg_right_transfer_region_indices = range(
            transfer_params['right_cg_window_start'],
            transfer_params['right_cg_window_end'])
    fg_right_transfer_region_indices = range(
            transfer_params['right_fg_window_start'],
            transfer_params['right_fg_window_end'], ref_factor)
    right_transfer_region_indices = zip(cg_right_transfer_region_indices,
            fg_right_transfer_region_indices)
    for cg_i, fg_i in right_transfer_region_indices:
        cg_e = coarse_grid.es[cg_i]
        cg_b = coarse_grid.bs[cg_i]

        for j in range(ref_factor):
            fg_e = fine_grid.es[fg_i + j]
            fg_b = fine_grid.bs[fg_i + j]

            right_transfer_region_energy += (
                    ((cg_b + fg_b)**2 + (cg_e + fg_e)**2) * fine_grid.dx)

    fine_grid_indices = (
            list(range(transfer_params['left_fg_window_end'],
                transfer_params['right_fg_window_start'])) +
            [0, 1, fine_grid.size - 2, fine_grid.size - 1])
    for i in fine_grid_indices:
        e = fine_grid.es[i]
        b = fine_grid.bs[i]
        fine_grid_energy += (e * e + b * b) * fine_grid.dx

    return (coarse_grid_energy, fine_grid_energy, left_transfer_region_energy,
            right_transfer_region_energy)

def plot_energies(coarse_grid_energies, fine_grid_energies,
        left_transfer_region_energies, right_transfer_region_energies,
        output_dir):
    plt.clf()
    default_fig_size = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(8, 12)

    iterations = [defs.ENERGY_OUTPUT_PERIOD * i
            for i in range(len(fine_grid_energies))]
    plt.plot(iterations, coarse_grid_energies, 'b', label='coarse')
    plt.plot(iterations, fine_grid_energies, 'r', label='fine')
    plt.plot(iterations, left_transfer_region_energies, 'c',
            label='left transfer')
    plt.plot(iterations, right_transfer_region_energies, 'm',
            label='right transfer')

    total_energies = [cg + fg + ltr + rtr  for (cg, fg, ltr, rtr) in
            zip(coarse_grid_energies, fine_grid_energies,
                left_transfer_region_energies, right_transfer_region_energies)]
    plt.plot(iterations, total_energies, 'k', label='total')

    plt.grid()
    plt.xlabel('iterations')
    plt.ylabel(r'$\propto$ energy')
    plt.legend(loc='best')
    plt.savefig(output_dir + os.sep + 'energy.pdf')

    plt.gcf().set_size_inches(default_fig_size)


def simulate(ref_factor, output_dir):
    # coarse grid
    coarse_grid = Grid(defs.COARSE_GRID_SIZE, defs.x0, defs.dx, defs.dt)
    coarse_grid.add_pml(Pml(defs.PML_SIZE, 1, coarse_grid))
    coarse_grid.add_pml(Pml(defs.COARSE_GRID_SIZE - defs.PML_SIZE - 1,
                            defs.COARSE_GRID_SIZE - 2, coarse_grid))

    # fine grid
    fine_grid_start = defs.FINE_GRID_START
    fine_grid_end = fine_grid_start + defs.FINE_GRID_SIZE
    fine_x0 = (defs.x0 + (fine_grid_start - defs.FFT_WINDOW_SIZE) * defs.dx -
            2 * defs.dx / ref_factor)
    fine_fft_window_size = defs.FFT_WINDOW_SIZE * ref_factor
    fine_grid_size = (defs.FINE_GRID_SIZE * ref_factor +
        2 * (2 + fine_fft_window_size))

    fine_grid = Grid(fine_grid_size, fine_x0, defs.dx / ref_factor,
                     defs.dt / ref_factor)

    add_source_current_to_b, add_source_current_to_e = get_field_generator(
        coarse_grid)

    # pack grids and indices into dicts for ease of passing into functions
    transfer_params = {
        'left_cg_window_start': fine_grid_start - defs.FFT_WINDOW_SIZE,
        'left_cg_window_end': fine_grid_start,
        'left_fg_window_start': 2,
        'left_fg_window_end': 2 + fine_fft_window_size,

        'right_cg_window_start': fine_grid_end,
        'right_cg_window_end': fine_grid_end + defs.FFT_WINDOW_SIZE,
        'right_fg_window_start': fine_grid_size - 2 - fine_fft_window_size,
        'right_fg_window_end': fine_grid_size - 2
    }

    left_transfer = Transfer(
        coarse_grid.slice(transfer_params['left_cg_window_start'],
                          transfer_params['left_cg_window_end']),
        fine_grid.slice(transfer_params['left_fg_window_start'],
                        transfer_params['left_fg_window_end']),
        fine_to_the_right=True)
    right_transfer = Transfer(
        coarse_grid.slice(transfer_params['right_cg_window_start'],
                          transfer_params['right_cg_window_end']),
        fine_grid.slice(transfer_params['right_fg_window_start'],
                        transfer_params['right_fg_window_end']),
        fine_to_the_right=False)

    coarse_grid_energies = []
    fine_grid_energies = []
    left_transfer_region_energies = []
    right_transfer_region_energies = []

    sys.stdout.write('iteration ')
    for t in range(defs.ITERATIONS):
        t_str = '{} out of {}'.format(t, defs.ITERATIONS)
        sys.stdout.write(t_str)
        sys.stdout.flush()

        cg_skip = lambda i: i >= fine_grid_start + 2 and i < fine_grid_end - 2

        update_half_b(coarse_grid, skip=cg_skip, first_half=False)
        add_source_current_to_b(t * defs.dt)

        update_e(coarse_grid, skip=cg_skip)
        add_source_current_to_e((t + 0.5) * defs.dt)

        update_half_b(coarse_grid, skip=cg_skip, first_half=True)

        for _ in range(ref_factor):
            update_half_b(fine_grid)
            update_e(fine_grid)
            update_half_b(fine_grid)

        left_transfer.perform()
        right_transfer.perform()

        if t % defs.OUTPUT_PERIOD == 0:
            build_plot(coarse_grid, fine_grid, t // defs.OUTPUT_PERIOD,
                       output_dir)

        if t % defs.ENERGY_OUTPUT_PERIOD == 0:
            cg_en, fg_en, ltr_en, rtr_en = calculate_energy(
                    coarse_grid, fine_grid, ref_factor, transfer_params)
            coarse_grid_energies.append(cg_en)
            fine_grid_energies.append(fg_en)
            left_transfer_region_energies.append(ltr_en)
            right_transfer_region_energies.append(rtr_en)


        sys.stdout.write('\b' * len(t_str))

    sys.stdout.write('\n')
    plot_energies(
            coarse_grid_energies, fine_grid_energies,
            left_transfer_region_energies, right_transfer_region_energies,
            output_dir)
    with open(output_dir + os.sep + 'energies.txt', 'wt') as enfile:
        iter = 0
        for cg_en, fg_en, ltr_en, rtr_en in zip(
                coarse_grid_energies,
                fine_grid_energies,
                left_transfer_region_energies,
                right_transfer_region_energies):
            print('{0} {1:.15g} {2:.15g} {3:.15g} {4:.15g}'.format(
                iter, cg_en, fg_en, ltr_en, rtr_en), file=enfile)
            iter += defs.ENERGY_OUTPUT_PERIOD


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_factor", help="fine grid refinement factor",
            type=int)
    parser.add_argument("output_dir", help="output directory")
    args = parser.parse_args()

    simulate(args.ref_factor, args.output_dir)
