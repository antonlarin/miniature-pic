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
    def __init__(self, coarse_grid_view, fine_grid_view, fine_to_the_right):
        self.coarse_grid_view = coarse_grid_view
        self.fine_grid_view = fine_grid_view
        self.cg2fg_direction = 1 if fine_to_the_right else -1
        self.fg2cg_direction = -1 * self.cg2fg_direction

        self.coarse_buffer_size = coarse_grid_view.size
        self.fine_buffer_size = fine_grid_view.size
        self.fspace_interpolation_factor = (self.fine_buffer_size /
                                            self.coarse_buffer_size)

        common_coeff = defs.TRANSFER_RATIO * 0.5
        self.coarse_mask = [
            common_coeff * defs.transfer_mask(i / self.coarse_buffer_size)
            for i in range(self.coarse_buffer_size)]
        self.fine_mask = [
            common_coeff * defs.transfer_mask(i / self.fine_buffer_size)
            for i in range(self.fine_buffer_size)]


    def get_coarse_to_fine_deltas(self):
        coarse_buffer = (self.coarse_mask *
                         (self.cg2fg_direction * self.coarse_grid_view.bs +
                          self.coarse_grid_view.es))

        coarse_fspace_buffer = np.fft.fft(coarse_buffer)
        fine_fspace_buffer = (self.fspace_interpolation_factor *
                              resize_fspace_buffer(coarse_fspace_buffer,
                                                   self.fine_buffer_size))
        fine_buffer = np.fft.ifft(fine_fspace_buffer).real

        return (-self.cg2fg_direction * coarse_buffer, -coarse_buffer,
                self.cg2fg_direction * fine_buffer, fine_buffer)


    def get_fine_to_coarse_deltas(self):
        fine_buffer = (self.fine_mask *
                       (self.fg2cg_direction * self.fine_grid_view.bs +
                        self.fine_grid_view.es))

        fine_fspace_buffer = np.fft.fft(fine_buffer)
        coarse_fspace_buffer = (resize_fspace_buffer(fine_fspace_buffer,
                                                     self.coarse_buffer_size) /
                                self.fspace_interpolation_factor)
        coarse_buffer = np.fft.ifft(coarse_fspace_buffer).real

        return (-self.fg2cg_direction * fine_buffer, -fine_buffer,
                self.fg2cg_direction * coarse_buffer, coarse_buffer)


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
        self.fine_grid_view.bs += (fg2cg_source_delta_b +
                                   cg2fg_target_delta_b)
        self.fine_grid_view.es += (fg2cg_source_delta_e +
                                   cg2fg_target_delta_e)
