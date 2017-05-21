from __future__ import division

import numpy as np

import defs


def resize_fspace_buffer(buffer, new_size):
    '''Pad or trim ndarray to preserve FFT freq structure

    Numpy FFT routines arrange output so that over the first
    half of the array the frequencies are positive and they increase
    and over the last half they are negative and increase. This
    function resizes the array in such a way that the frequencies
    with higher absolute values are affected. When padding the array
    zeros are inserted in the middle, and when trimming it, values
    are deleted from the middle.
    '''
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


def lagrange3_middle(f0, f1, f2, f3):
    '''Cubic Lagrangian interpolation at the center of the stencil
    '''
    return (-f0 + 9*f1 + 9*f2 - f3) / 16


def fill_raw_buffer(src_grid, src_indices, direction_coeff):
    '''Compute the wave component with a specific travel direction

    0.5 * (E + H) for L2R-travelling wave and 0.5 * (E - H) for
    R2L-travelling wave. The direction is specified via
    direction_coeff, which is a sign in front of H in the
    formulae above.
    Yee grid allows the resulting buffer to have twice as many
    values as there are cells in the specified region, but it
    requires using interpolation for both electric and magnetic
    fields.
    '''
    cell_count = len(src_indices)
    buffer_size = 2 * cell_count
    transfer_buffer = np.zeros(buffer_size)

    for i in src_indices:
        local_i = i - src_indices[0]

        transfer_buffer[2 * local_i] = (
            direction_coeff * src_grid.bs[i] + lagrange3_middle(
                src_grid.es[i - 2],
                src_grid.es[i - 1],
                src_grid.es[i],
                src_grid.es[i + 1]))

        transfer_buffer[2 * local_i + 1] = (
            src_grid.es[i] + direction_coeff * lagrange3_middle(
                src_grid.bs[i - 1],
                src_grid.bs[i],
                src_grid.bs[i + 1],
                src_grid.bs[i + 2]))

    return transfer_buffer


class Transfer(object):
    '''A pair of transfers between a coarser and a finer grid

    Transfers in a pair are interconnected: it's necessary to first
    compute the transferred values for both grids and then to
    modify the grids' contents. Instances of this class serve 2 purposes:
    they precompute and store useful intermediate values and
    each iteration produce the deltas that need to be substracted from
    the source grids and added to the target grids. The deltas already
    contain the sign, so Transfer.perform can simply add all of them together.
    '''
    def __init__(self, coarse_grid, fine_grid, transfer_params, fine_to_the_right):
        self.coarse_grid = coarse_grid
        self.fine_grid = fine_grid
        for key in transfer_params:
            setattr(self, key, transfer_params[key])

        self.cg2fg_direction = 1 if fine_to_the_right else -1
        self.fg2cg_direction = -1 * self.cg2fg_direction

        self.cg_indices = range(self.cg_window_start, self.cg_window_end)
        self.fg_indices = range(self.fg_window_start, self.fg_window_end)

        self.coarse_buffer_size = 2 * len(self.cg_indices)
        self.fine_buffer_size = 2 * len(self.fg_indices)

        common_coeff = defs.TRANSFER_RATIO * 0.5
        self.coarse_mask = [
            common_coeff * defs.transfer_mask(i / self.coarse_buffer_size)
            for i in range(self.coarse_buffer_size)]
        self.fine_mask = [
            common_coeff * defs.transfer_mask(i / self.fine_buffer_size)
            for i in range(self.fine_buffer_size)]


    def get_coarse_to_fine_deltas(self):
        raw_coarse_buffer = fill_raw_buffer(
            self.coarse_grid, self.cg_indices, self.cg2fg_direction)
        coarse_buffer = self.coarse_mask * raw_coarse_buffer

        coarse_fspace_buffer = np.fft.fft(coarse_buffer)
        fine_fspace_buffer = self.ref_factor * resize_fspace_buffer(
            coarse_fspace_buffer, self.fine_buffer_size)
        fine_buffer = np.fft.ifft(fine_fspace_buffer).real

        return (-self.cg2fg_direction * coarse_buffer[::2], -coarse_buffer[1::2],
                self.cg2fg_direction * fine_buffer[::2], fine_buffer[1::2])


    def get_fine_to_coarse_deltas(self):
        raw_fine_buffer = fill_raw_buffer(
            self.fine_grid, self.fg_indices, self.fg2cg_direction)
        fine_buffer = self.fine_mask * raw_fine_buffer

        fine_fspace_buffer = np.fft.fft(fine_buffer)
        coarse_fspace_buffer = (
            resize_fspace_buffer(fine_fspace_buffer,
                                 self.coarse_buffer_size) / self.ref_factor)
        coarse_buffer = np.fft.ifft(coarse_fspace_buffer).real

        return (-self.fg2cg_direction * fine_buffer[::2], -fine_buffer[1::2],
                self.fg2cg_direction * coarse_buffer[::2], coarse_buffer[1::2])


    def perform(self):
        (cg2fg_source_delta_b,
         cg2fg_source_delta_e,
         cg2fg_target_delta_b,
         cg2fg_target_delta_e) = self.get_coarse_to_fine_deltas()
        (fg2cg_source_delta_b,
         fg2cg_source_delta_e,
         fg2cg_target_delta_b,
         fg2cg_target_delta_e) = self.get_fine_to_coarse_deltas()

        self.coarse_grid.bs[self.cg_indices] += (
            cg2fg_source_delta_b + fg2cg_target_delta_b)
        self.coarse_grid.es[self.cg_indices] += (
            cg2fg_source_delta_e + fg2cg_target_delta_e)
        self.fine_grid.bs[self.fg_indices] += (
            fg2cg_source_delta_b + cg2fg_target_delta_b)
        self.fine_grid.es[self.fg_indices] += (
            fg2cg_source_delta_e + cg2fg_target_delta_e)
