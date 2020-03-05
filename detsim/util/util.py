import numpy  as np
import pandas as pd

from typing import  List
from typing import Tuple

from invisible_cities.io      .mcinfo_io import get_sensor_binning
from invisible_cities.database.load_db   import            DataPMT
from invisible_cities.database.load_db   import           DataSiPM


def trigger_times(trigger_indx: List[int] ,
                  event_time  :      float,
                  time_bins   : np.ndarray) -> List[int]:

    return [event_time + time_bins[trg] for trg in trigger_indx]


def first_and_last_times(sensor_bin_times: np.ndarray) -> Tuple:
    min_time = sensor_bin_times[ 0]
    max_time = sensor_bin_times[-1] + np.diff(sensor_bin_times)[-1]
    return min_time, max_time


def sensor_order(pmt_wfs    : pd.Series,
                 sipm_wfs   : pd.Series,
                 detector_db:       str,
                 run_number :       int) -> Tuple:
    pmts     = DataPMT (detector_db, run_number).SensorID
    sipms    = DataSiPM(detector_db, run_number).SensorID
    pmt_ord  = pmts [ pmts.isin( pmt_wfs.index.tolist())].index
    sipm_ord = sipms[sipms.isin(sipm_wfs.index.tolist())].index
    return pmt_ord, sipm_ord


def get_no_sensors(detector_db: str, run_number: int) -> Tuple:
    npmt  = DataPMT (detector_db, run_number).shape[0]
    nsipm = DataSiPM(detector_db, run_number).shape[0]
    return npmt, nsipm


def pmt_and_sipm_bin_width(file_name: str) -> Tuple:
    ## Temp function to return pmt and sipm bin widths
    sns_bins = get_sensor_binning(file_name)
    pmt_wid  = sns_bins.bin_width[sns_bins.index.str.contains( 'Pmt')].iloc[0]
    sipm_wid = sns_bins.bin_width[sns_bins.index.str.contains('SiPM')].iloc[0]
    return pmt_wid, sipm_wid
