from __future__ import division
import math

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


class Transfer:
    '''A pair of transfers between a coarser and a finer grid

    Transfers in a pair are interconnected: it's necessary to first
    compute the transferred values for both grids and then to
    modify the grids' contents. Instances of this class serve 2 purposes:
    they precompute and store useful intermediate values and
    each iteration produce the deltas that need to be substracted from
    the source grids and added to the target grids. The deltas already
    contain the sign, so Transfer.perform can simply add all of them together.
    '''
    def __init__(self, coarse_grid_view, fine_grid_view, fine_to_the_right):
        self.coarse_grid_view = coarse_grid_view
        self.fine_grid_view = fine_grid_view
        self.cg2fg_direction = 1 if fine_to_the_right else -1
        self.fg2cg_direction = -1 * self.cg2fg_direction

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

        source_buffer_fspace = 0.5 * (self.cg2fg_direction * bs_fspace +
                                      es_fspace)
        target_buffer_fspace = (self.fspace_interpolation_factor *
                                resize_fspace_buffer(source_buffer_fspace,
                                                     self.fine_buffer_size))

        source_delta_b = (-self.cg2fg_direction *
                          np.fft.ifft(source_buffer_fspace).real)
        source_delta_e = -np.fft.ifft(
            np.multiply(source_buffer_fspace,
                        self.coarse_e_inverse_shifts)).real

        target_delta_b = (self.cg2fg_direction *
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

        source_buffer_fspace = 0.5 * (self.fg2cg_direction * bs_fspace +
                                      es_fspace)
        target_buffer_fspace = (
            resize_fspace_buffer(source_buffer_fspace,
                                 self.coarse_buffer_size) /
            self.fspace_interpolation_factor)

        source_delta_b = (-self.fg2cg_direction *
                          np.fft.ifft(source_buffer_fspace).real)
        source_delta_e = -np.fft.ifft(
            np.multiply(source_buffer_fspace,
                        self.fine_e_inverse_shifts)).real

        target_delta_b = (self.fg2cg_direction *
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
