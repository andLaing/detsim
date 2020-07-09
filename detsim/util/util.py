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


def first_and_last_times(pmt_wfs    : pd.Series,
                         sipm_wfs   : pd.Series,
                         pmt_binwid : float    ,
                         sipm_binwid: float    ) -> Tuple:
    min_time  = min(pmt_wfs.time.min(), sipm_wfs.time.min())
    max_time  = max(pmt_wfs.time.max(), sipm_wfs.time.max())
    max_time += min(pmt_binwid        ,         sipm_binwid)
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

def order_sensors(detector_db: str, run_number : int,
                  n_pmt      : int, length_pmt : int,
                  n_sipm     : int, length_sipm: int) -> Callable:
    """
    Casts the event sensor info into the correct order
    adding zeros for sensors which didn't see any signal.
    """
    pmt_ids    = DataPMT (detector_db, run_number).SensorID
    sipm_ids   = DataSiPM(detector_db, run_number).SensorID
    pmt_shape  = (n_pmt , length_pmt )
    sipm_shape = (n_sipm, length_sipm)
    def ordering(sensor_order : pd.Int64Index,
                 sensor_resp  : np.ndarray   ,
                 sensor_shape : Tuple        ) -> np.ndarray:
        sensors = np.zeros(sensor_shape, np.int)
        sensors[sensor_order] = sensor_resp
        return sensors
        
    def order_and_pad(pmt_resp      : pd.Series  ,
                      sipm_resp     : pd.Series  ,
                      evt_buffers : List[Tuple]) -> List[Tuple]:
        pmt_ord  = pmt_ids [ pmt_ids.isin( pmt_resp.index.tolist())].index
        sipm_ord = sipm_ids[sipm_ids.isin(sipm_resp.index.tolist())].index

        return [(ordering(pmt_ord , pmts , pmt_shape ),
                 ordering(sipm_ord, sipms, sipm_shape))
                    for pmts, sipms in evt_buffers]
    return order_and_pad


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
