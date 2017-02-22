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
            if not grid.in_pml(i):
                grid.bs[i] = (A * grid.bs[i] +
                        0.5 * B * (grid.es[i - 1] - grid.es[i]))
            elif not first_half:
                grid.bs[i] = A * grid.bs[i] + B * (grid.es[i - 1] - grid.es[i])


# field generation
def get_field_generator(coarse_grid):
    _fieldgen_idx = defs.PML_SIZE + 1;
    left_e_x = coarse_grid.x_of(_fieldgen_idx + 0.5)
    right_e_x = coarse_grid.x_of(defs.COARSE_GRID_SIZE - _fieldgen_idx - 0.5)
    def generate_b(t):
        # left
        b_source(coarse_grid, _fieldgen_idx, defs.left_e(left_e_x, t),
                'right')

        # right
        b_source(coarse_grid, defs.COARSE_GRID_SIZE - _fieldgen_idx,
                defs.right_e(right_e_x, t), 'left')

    left_b_x = coarse_grid.x_of(_fieldgen_idx)
    right_b_x = coarse_grid.x_of(defs.COARSE_GRID_SIZE - _fieldgen_idx)
    def generate_e(t):
        # left
        e_source(coarse_grid, _fieldgen_idx, defs.left_b(left_b_x, t),
                'right')

        # right
        e_source(coarse_grid, defs.COARSE_GRID_SIZE - _fieldgen_idx - 1,
                defs.right_b(right_b_x, t), 'left')

    return (generate_b, generate_e)


# transfers
def resize_fspace_buffer(buffer, new_size):
    if new_size >= len(buffer):
        padding_size = new_size - len(buffer)
        pad_items = np.zeros(padding_size, dtype=complex)
        resized_buffer = np.insert(buffer, -(len(buffer) // 2), pad_items)
    else:
        nonnegative_freq_count = (new_size + 1) // 2
        negative_freq_count = new_size // 2
        resized_buffer = np.delete(buffer,
                range(nonnegative_freq_count,
                    len(buffer) - negative_freq_count))

    return resized_buffer


class Transfer:
    def __init__(self, coarse_grid_view, fine_grid_view, fine_to_the_right):
        self.coarse_grid_view = coarse_grid_view
        self.fine_grid_view = fine_grid_view
        self.direction_coeff = 1 if fine_to_the_right else -1

        self.coarse_buffer_size = coarse_grid_view.size
        self.fine_buffer_size = fine_grid_view.size
        self.fspace_interpolation_factor = (
            self.fine_buffer_size / self.coarse_buffer_size)

        self.coarse_b_mask = [
            defs.transfer_mask(i / self.coarse_buffer_size)
            for i in range(self.coarse_buffer_size)]
        self.coarse_e_mask = [
            defs.transfer_mask((i + 0.5) / self.coarse_buffer_size)
            for i in range(self.coarse_buffer_size)]
        self.fine_b_mask = [
            defs.transfer_mask(i / self.fine_buffer_size)
            for i in range(self.fine_buffer_size)]
        self.fine_e_mask = [
            defs.transfer_mask((i + 0.5) / self.fine_buffer_size)
            for i in range(self.fine_buffer_size)]

        self.coarse_e_shifts = np.exp(
            -1j * math.pi * np.fft.fftfreq(self.coarse_buffer_size))
        self.coarse_e_inverse_shifts = 1 / self.coarse_e_shifts
        self.fine_e_shifts = np.exp(
            -1j * math.pi * np.fft.fftfreq(self.fine_buffer_size))
        self.fine_e_inverse_shifts = 1 / self.fine_e_shifts


    def get_coarse_to_fine_deltas(self):
        bs_masked = (
            defs.TRANSFER_RATIO *
            np.multiply(self.coarse_grid_view.bs, self.coarse_b_mask))
        es_masked = (
            defs.TRANSFER_RATIO *
            np.multiply(self.coarse_grid_view.es, self.coarse_e_mask))

        bs_fspace = np.fft.fft(bs_masked)
        es_fspace = np.multiply(np.fft.fft(es_masked), self.coarse_e_shifts)

        source_buffer_fspace = 0.5 * (self.direction_coeff * bs_fspace +
                                      es_fspace)
        target_buffer_fspace = (self.fspace_interpolation_factor *
                                resize_fspace_buffer(source_buffer_fspace,
                                                     self.fine_buffer_size))

        source_delta_b = (-self.direction_coeff *
                          np.fft.ifft(source_buffer_fspace).real)
        source_delta_e = -np.fft.ifft(
            np.multiply(source_buffer_fspace,
                        self.coarse_e_inverse_shifts)).real

        target_delta_b = (self.direction_coeff *
                          np.fft.ifft(target_buffer_fspace).real)
        target_delta_e = np.fft.ifft(
            np.multiply(target_buffer_fspace,
                        self.fine_e_inverse_shifts)).real

        return (source_delta_b, source_delta_e,
                target_delta_b, target_delta_e)


    def get_fine_to_coarse_deltas(self):
        bs_masked = (
            defs.TRANSFER_RATIO *
            np.multiply(self.fine_grid_view.bs, self.fine_b_mask))
        es_masked = (
            defs.TRANSFER_RATIO *
            np.multiply(self.fine_grid_view.es, self.fine_e_mask))

        bs_fspace = np.fft.fft(bs_masked)
        es_fspace = np.multiply(np.fft.fft(es_masked), self.fine_e_shifts)

        source_buffer_fspace = 0.5 * (-self.direction_coeff * bs_fspace +
                                      es_fspace)
        target_buffer_fspace = (
            resize_fspace_buffer(source_buffer_fspace,
                                 self.coarse_buffer_size) /
            self.fspace_interpolation_factor)

        source_delta_b = (self.direction_coeff *
                          np.fft.ifft(source_buffer_fspace).real)
        source_delta_e = -np.fft.ifft(
            np.multiply(source_buffer_fspace,
                        self.fine_e_inverse_shifts)).real

        target_delta_b = (-self.direction_coeff *
                          np.fft.ifft(target_buffer_fspace).real)
        target_delta_e = np.fft.ifft(
            np.multiply(target_buffer_fspace,
                        self.coarse_e_inverse_shifts)).real

        return (source_delta_b, source_delta_e,
                target_delta_b, target_delta_e)


    def perform(self):
        (cg2fg_source_delta_b,
         cg2fg_source_delta_e,
         cg2fg_target_delta_b,
         cg2fg_target_delta_e) = self.get_coarse_to_fine_deltas()
        (fg2cg_source_delta_b,
         fg2cg_source_delta_e,
         fg2cg_target_delta_b,
         fg2cg_target_delta_e) = self.get_fine_to_coarse_deltas()

        self.coarse_grid_view.bs += (cg2fg_source_delta_b +
                                     fg2cg_target_delta_b)
        self.coarse_grid_view.es += (cg2fg_source_delta_e +
                                     fg2cg_target_delta_e)
        self.fine_grid_view.bs += (cg2fg_target_delta_b +
                                   fg2cg_source_delta_b)
        self.fine_grid_view.es += (cg2fg_target_delta_e +
                                   fg2cg_source_delta_e)


# utility
def build_plot(coarse_grid, fine_grid, pic_idx, output_dir):
    xs = [coarse_grid.x_of(i) for i in range(len(coarse_grid.bs))]
    fine_xs = [fine_grid.x_of(i) for i in range(len(fine_grid.bs))]

    plt.clf()
    plt.gcf().set_size_inches(8, 10)
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
    plt.ylim(1e-10, 1)
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
    plt.ylim(1e-10, 1)
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

    total_energies = np.asarray([cg + fg + ltr + rtr
                                 for (cg, fg, ltr, rtr) in
                                 zip(coarse_grid_energies,
                                     fine_grid_energies,
                                     left_transfer_region_energies,
                                     right_transfer_region_energies)])
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
    fine_grid_start = (defs.COARSE_GRID_SIZE - defs.FINE_GRID_SIZE) // 2
    fine_grid_end = fine_grid_start + defs.FINE_GRID_SIZE
    fine_x0 = (defs.x0 + (fine_grid_start - defs.FFT_WINDOW_SIZE) * defs.dx -
            2 * defs.dx / ref_factor)
    fine_fft_window_size = defs.FFT_WINDOW_SIZE * ref_factor
    fine_grid_size = (defs.FINE_GRID_SIZE * ref_factor +
        2 * (2 + fine_fft_window_size))

    fine_grid = Grid(fine_grid_size, fine_x0, defs.dx / ref_factor,
                     defs.dt / ref_factor)

    # pack grids and indices into dicts for ease of passing into functions
    transfer_params = {
            'ref_factor': ref_factor,

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

    generate_b, generate_e = get_field_generator(coarse_grid)

    coarse_grid_energies = []
    fine_grid_energies = []
    left_transfer_region_energies = []
    right_transfer_region_energies = []

    sys.stdout.write('iteration ')
    for t in range(defs.ITERATIONS):
        t_str = '{}'.format(t)
        sys.stdout.write(t_str)
        sys.stdout.flush()

        cg_skip = lambda i: i >= fine_grid_start + 2 and i < fine_grid_end - 2

        # update fields on coarse grid
        update_b(coarse_grid, False, skip=cg_skip)
        generate_b(t * coarse_grid.dt)

        update_e(coarse_grid, skip=cg_skip)
        generate_e((t + 0.5) * coarse_grid.dt)

        update_b(coarse_grid, True, skip=cg_skip)

        # update fields on fine grid
        for _ in range(ref_factor):
            update_b(fine_grid, False)
            update_e(fine_grid)
            update_b(fine_grid, True)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_factor", help="fine grid refinement factor",
            type=int)
    parser.add_argument("output_dir", help="output directory")
    args = parser.parse_args()

    simulate(args.ref_factor, args.output_dir)
