"""
Module taking in nexus full simulation sensitive detector info and
positioning the signal according to a given buffer size, pre-trigger
and simple 'trigger' conditions which allow for the positioning of
the signal within the buffers.
"""

import   os
import json
import  sys

import numpy  as np
import pandas as pd
import tables as tb

from glob      import    glob
from functools import partial
from functools import   wraps

from detsim.io        .hdf5_io          import        buffer_writer
from detsim.io        .hdf5_io          import         load_sensors
from detsim.io        .hdf5_io          import        save_run_info
from detsim.simulation.buffer_functions import    calculate_buffers
from detsim.simulation.buffer_functions import        signal_finder
from detsim.simulation.buffer_functions import            wf_binner
from detsim.util      .util             import first_and_last_times
from detsim.util      .util             import       get_no_sensors
from detsim.util      .util             import         sensor_order
from detsim.util      .util             import        trigger_times

from invisible_cities.core.configure         import          configure
from invisible_cities.core.system_of_units_c import              units
from invisible_cities.io  .mcinfo_io         import get_sensor_binning
from invisible_cities.io  .mcinfo_io         import     mc_info_writer
from invisible_cities.reco                   import      tbl_functions as tbl

from invisible_cities.dataflow          import dataflow as fl
from invisible_cities.dataflow.dataflow import     fork
from invisible_cities.dataflow.dataflow import     pipe
from invisible_cities.dataflow.dataflow import     push


def position_signal(conf):

    files_in      = glob(os.path.expandvars(conf.files_in))
    file_out      =      os.path.expandvars(conf.file_out)
    detector_db   =                         conf.detector_db
    run_number    =                     int(conf.run_number)
    max_time      =                     int(conf.max_time)
    buffer_length =                   float(conf.buffer_length)
    pre_trigger   =                   float(conf.pre_trigger)
    trg_threshold =                   float(conf.trg_threshold)
    compression   =                         conf.compression

    npmt, nsipm        = get_no_sensors(detector_db, run_number)
    pmt_wid, sipm_wid  = get_sensor_binning(files_in[0])
    nsamp_pmt          = int(buffer_length * units.mus /  pmt_wid)
    nsamp_sipm         = int(buffer_length * units.mus / sipm_wid)

    bin_calculation    = wf_binner(max_time)
    bin_pmt_wf         = fl.map(bin_calculation,
                                args = ("pmt_wfs" ,  "pmt_binwid"),
                                out  = ("pmt_bins", "pmt_bin_wfs"))

    extract_minmax     = fl.map(first_and_last_times,
                                args = "pmt_bins",
                                out  = ("min_time", "max_time"))

    bin_sipm_wf        = fl.map(bin_calculation,
                                args = ("sipm_wfs", "sipm_binwid",
                                        "min_time",    "max_time") ,
                                out  = ("sipm_bins", "sipm_bin_wfs"))

    sensor_order_      = fl.map(partial(sensor_order,
                                        detector_db = detector_db,
                                        run_number  =  run_number),
                                args = ("pmt_bin_wfs", "sipm_bin_wfs"),
                                out  = ("pmt_ord", "sipm_ord"))

    signal_finder_     = fl.map(signal_finder(buffer_length,
                                              pmt_wid, trg_threshold),
                                args = "pmt_bin_wfs",
                                out  = "pulses")

    event_times        = fl.map(trigger_times,
                                args = ("pulses", "timestamp", "pmt_bins"),
                                out  = "evt_times")

    calculate_buffers_ = fl.map(calculate_buffers(buffer_length, pre_trigger,
                                                  pmt_wid      ,    sipm_wid),
                                args = ("pulses",
                                        "pmt_bins" ,  "pmt_bin_wfs",
                                        "sipm_bins", "sipm_bin_wfs"),
                                out  = "buffers")

    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:

        write_mc       = fl.sink(mc_info_writer(h5out),
                                 args = ("mc", "evt"))
        buffer_writer_ = fl.sink(buffer_writer(h5out                  ,
                                               n_sens_eng = npmt      ,
                                               n_sens_trk = nsipm     ,
                                               length_eng = nsamp_pmt ,
                                               length_trk = nsamp_sipm),
                                 args = ("evt", "pmt_ord", "sipm_ord",
                                         "evt_times", "buffers"))

        save_run_info(h5out, run_number)
        return push(source = load_sensors(files_in, detector_db, run_number),
                    pipe   = pipe(bin_pmt_wf          ,
                                  extract_minmax      ,
                                  bin_sipm_wf         ,
                                  sensor_order_       ,
                                  signal_finder_      ,
                                  event_times         ,
                                  calculate_buffers_  ,
                                  fork(buffer_writer_,
                                       write_mc      )))



if __name__ == "__main__":
    conf = configure(sys.argv).as_namespace
    position_signal(conf)
