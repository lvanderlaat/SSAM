import numpy as np
import pandas as pd
from obspy_tools.filter import bin_smooth_spectrum
from obspy_tools.obspy2numpy import tr2windowed_data
from obspy.core.util.misc import get_window_times

def ssam(tr, window_length, freq_start, freq_end, bin_width, mode):
    """
    Calculates SSAM for a Trace using FFT, implement Yule-Walker
    Parameters
    ----------
    tr : obspy Trace
        Trace
    window_length : float
        Window length in seconds, this determines the pixel size
    freq_start : float
        Lower frequency to cut the spectra
    freq_end : float
        Upper frequency to cut the spectra
    bin_width : float
        Frequency bin width [Hz]
    mode : str
        'mean', 'median' or 'sum_sqrt'
    Returns
    -------
    fft : np array
        fft.shape = (n_windows, n_bins)
    time : np 1d array
        Array containing the times as obspy UTCDatetime objects of window
        centers
    """
    data_windowed, time = tr2windowed_data_exp(tr, window_length)

    fft = np.abs(np.fft.rfft(data_windowed))
    freq = np.fft.rfftfreq(window, tr.stats.delta)

    fft = np.apply_along_axis(bin_smooth_spectrum, axis=1, fft, freq,
                             freq_start, freq_end, bin_width, mode)
    return fft, time
