import numpy  as np
import pandas as pd

from invisible_cities.reco.peak_functions    import indices_and_wf_above_threshold
from invisible_cities.reco.peak_functions    import                 split_in_peaks
from invisible_cities.evm .event_model       import                       Waveform
from invisible_cities.core.system_of_units_c import                          units

from typing    import  Callable
from typing    import Generator
from typing    import      List
from typing    import   Mapping
from typing    import     Tuple

from functools import     wraps


@wraps(np.histogram)
def weighted_histogram(data: pd.DataFrame, bins: np.ndarray) -> np.ndarray:
    return np.histogram(data.time, weights=data.charge, bins=bins)[0]


def padder(sensors: np.ndarray, padding: Tuple) -> np.ndarray:
    return np.apply_along_axis(np.pad, 1, sensors, padding, "constant")


def calculate_buffers(buffer_len: float, pre_trigger: float,
                      pmt_binwid: float, sipm_binwid: float) -> Callable:
    """
    Calculates the output buffers for all sensors
    based on a configured buffer length and pretrigger.

    buffer_len  : float
                  Length of buffer expected in mus
    pre_trigger : float
                  Time in buffer before identified signal in mus
    pmt_binwid  : float
                  Width in mus of PMT sample integration
    sipm_binwid : float
                  Width in mus of SiPM sample integration
    """

    pmt_buffer_samples  = int(buffer_len * units.mus /  pmt_binwid)
    sipm_buffer_samples = int(buffer_len * units.mus / sipm_binwid)
    sipm_pretrg         = int(pre_trigger * units.mus / sipm_binwid)
    sipm_postrg         = sipm_buffer_samples - sipm_pretrg
    pmt_pretrg_         = int(pre_trigger * units.mus / pmt_binwid)
    pmt_postrg_         = pmt_buffer_samples - pmt_pretrg_


    def sipm_trg_bin(sipm_bins: np.ndarray,
                     pmt_bins : np.ndarray) -> Callable[[int], int]:
        def get_sipm_bin(trigger: int) -> int:
            return np.where(sipm_bins <= pmt_bins[trigger])[0][-1]
        return get_sipm_bin


    def slice_generator(pmt_bins   : np.ndarray,
                        pmt_charge : np.ndarray,
                        sipm_bins  : np.ndarray,
                        sipm_charge: np.ndarray) -> Generator:

        npmt_bin  = len(pmt_bins)
        nsipm_bin = len(sipm_bins)
        sipm_trg  = sipm_trg_bin(sipm_bins, pmt_bins)

        def generate_slices(triggers: List) -> Tuple:

            for trg in triggers:
                trg_bin    = sipm_trg(trg)

                bin_corr   = (pmt_bins[trg] - sipm_bins[trg_bin]) / pmt_binwid
                pmt_pretrg = pmt_pretrg_ + int(bin_corr)
                pmt_postrg = pmt_postrg_ - int(bin_corr)

                pmt_pre    = 0       , trg - pmt_pretrg
                pmt_pos    = npmt_bin, trg + pmt_postrg
                pmt_sl     = slice(max(pmt_pre), min(pmt_pos))
                pmt_pad    = (int(-min(pmt_pre)),
                              int( max(0, pmt_pos[1] - npmt_bin + 1)))

                sipm_pre   = 0        , trg_bin - sipm_pretrg
                sipm_pos   = nsipm_bin, trg_bin + sipm_postrg
                sipm_sl    = slice(max(sipm_pre), min(sipm_pos))
                sipm_pad   = (int(-min(sipm_pre)),
                              int( max(0, sipm_pos[1] - nsipm_bin + 1)))

                yield ((pmt_charge [:,  pmt_sl],  pmt_pad),
                       (sipm_charge[:, sipm_sl], sipm_pad))
        return generate_slices


    def position_signal(triggers   :       List,
                        pmt_bins   : np.ndarray,
                        pmt_charge :  pd.Series,
                        sipm_bins  : np.ndarray,
                        sipm_charge:  pd.Series) -> List:

        slice_and_pad = slice_generator(pmt_bins                      ,
                                        np.array(pmt_charge .tolist()),
                                        sipm_bins                     ,
                                        np.array(sipm_charge.tolist()))

        return [(padder(*pmts), padder(*sipms))
                for pmts, sipms in slice_and_pad(triggers)]
    return position_signal


def wf_binner(max_buffer: int) -> Callable:
    """
    Returns a function to be used to convert the raw
    input Waveforms into data binned according to
    the bin width stored in the Waveforms, effectively
    padding with zeros inbetween the separate signals.

    max_buffer : float
        Maximum event time to be considered in nanoseconds
    """
    def bin_data(sensors  : pd.Series,
                 bin_width: float    ,
                 t_min    : float    ,
                 t_max    : float    ) -> Tuple:
        """
        Raw data binning function.

        sensors : List of Waveforms
            Should be sorted into one type/binning
        t_min : float
            Minimum time to be used to define bins.
        t_max : float
            As t_min but the maximum to be used
        """
        max_time = min(t_max, t_min + max_buffer)
        min_bin  = np.floor(t_min    / bin_width) * bin_width
        max_bin  = np.ceil (max_time / bin_width) * bin_width
        ## if t_min is None or t_max is None:
        ##     min_time = sensors.time.min()
        ##     max_time = min(sensors.time.max()   ,
        ##                    min_time + max_buffer)
        ##     min_bin  = np.floor(min_time / bin_width) * bin_width
        ##     max_bin  = np.floor(max_time / bin_width) * bin_width
        ##     max_bin += bin_width
        ## else:
        ##     ## Adjust according to bin_width
        ##     min_bin  = np.floor(t_min / bin_width) * bin_width
        ##     max_bin  = np.ceil (t_max / bin_width) * bin_width

        bins = np.arange(min_bin, max_bin, bin_width)

        bin_sensors = sensors.groupby('sensor_id').apply(weighted_histogram,
                                                         bins              )
        return bins, bin_sensors
    return bin_data


## !! to-do: clarify for non-pmt versions of next
def signal_finder(buffer_len   : float,
                  bin_width    : float,
                  bin_threshold:   int) -> Callable:
    """
    Decides where there is signal-like
    charge according to the configuration
    and the PMT sum in order to give
    a useful position for buffer selection

    buffer_len    : float
                    Configured buffer length in mus
    bin_width     : float
                    Sampling width for sensors
    bin_threshold : int
                    PE threshold for selection
    """

    stand_off = int(buffer_len * units.mus / bin_width)
    def find_signal(wfs: pd.Series) -> List[int]:

        eng_sum = wfs.sum(0)
        indices = indices_and_wf_above_threshold(eng_sum,
                                                 bin_threshold).indices
        ## Just using this and the stand_off for now
        ## taking first above sum threshold.
        ## !! To-do: make more robust with min int? or similar
        all_indx = split_in_peaks(indices, stand_off)
        return [pulse[0] for pulse in all_indx]
    return find_signal
