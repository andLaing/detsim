import numpy  as np
import pandas as pd
import tables as tb

from functools import     wraps
from typing    import  Callable
from typing    import Generator
from typing    import     Tuple
from typing    import      List

from invisible_cities.database           import                   load_db as  DB
from invisible_cities.io      .mcinfo_io import        get_sensor_binning
from invisible_cities.io      .mcinfo_io import load_mcsensor_response_df
from invisible_cities.io      .mcinfo_io import            load_mchits_df
from invisible_cities.io      .rwf_io    import                rwf_writer
from invisible_cities.reco               import             tbl_functions as tbl


class EventInfo(tb.IsDescription):
    """
    For the runInfo table to save event
    number and timestamp.
    #Additionally, saves the original nexus
    #event number
    """
    event_number = tb. Int32Col(shape=(), pos=0)
    timestamp    = tb.UInt64Col(shape=(), pos=1)
    #nexus_evt    = tb. Int32Col(shape=(), pos=2)


class EventMap(tb.IsDescription):
    """
    Maps the output event number and
    the original nexus event number
    NEEDS TO BE REVIEWED FOR INTEGRATION
    """
    event_number = tb.Int32Col(shape=(), pos=0)
    nexus_evt    = tb.Int32Col(shape=(), pos=1)


class RunInfo(tb.IsDescription):
    """
    Saves the run number in its own table
    This is expected by diomira.
    """
    run_number = tb.Int32Col(shape=(), pos=0)


def save_run_info(h5out     : tb.file.File,
                  run_number:          int) -> None:
    """
    Saves the run number used for the detsim
    format job in the format expected by the
    IC cities.

    h5out      : pytables file
                 The open output file
    run_number : int
                 The run number set in the config
    """

    try:
        run_table = getattr(h5out.root.Run, 'runInfo')
    except tb.NoSuchNodeError:
        try:
            run_group = getattr(h5out.root, 'Run')
            run_table = h5out.create_table(run_group, "runInfo", RunInfo,
                                           "Run number used in detsim")
        except tb.NoSuchNodeError:
            run_group = h5out.create_group(h5out.root, 'Run')
            run_table = h5out.create_table(run_group, "runInfo", RunInfo,
                                           "Run number used in detsim")
    row = run_table.row
    row["run_number"] = run_number
    row.append()


#def event_timestamp(h5in: tb.file.File) -> Callable:
def event_timestamp(file_name : str) ->Callable:
    """
    Returns a function iterator giving access
    to the next event's first hit time.
    This is used as the event timestamp.
    This is needed for the event information
    required by the IC cities.
    Generally set to zero in nexus but here for
    completeness.

    h5in : pytables file
           The input nexus hdf5 file.
    """
    ## New readers, no extents necessarily. Needs to be reviewed!
    hits     = load_mchits_df(file_name)
    time_it  = hits.groupby(level=0).time.min().iteritems()
    max_iter = len(hits.index.levels[0])
    def get_evt_timestamp() -> float:
        get_evt_timestamp.counter += 1
        if get_evt_timestamp.counter > max_iter:
            raise IndexError('No more events')
        return next(time_it)[1]
    get_evt_timestamp.counter = 0
    return get_evt_timestamp

    # The extents table saves the last hit index for each
    # event, we need the first so +1
    ## hit_indx   = (int(ext[2] + 1) for ext in h5in.root.MC.extents[:-1])
    ## first_hits = iter([0] + list(hit_indx))
    ## max_iter   = len(h5in.root.MC.extents)
    ## def get_evt_timestamp() -> float:
    ##     get_evt_timestamp.counter += 1
    ##     if get_evt_timestamp.counter > max_iter:
    ##         raise IndexError('No more events')
    ##     return h5in.root.MC.hits[next(first_hits)][2]
    ## get_evt_timestamp.counter = 0
    ## return get_evt_timestamp


@wraps(rwf_writer)
def buffer_writer(h5out, *,
                  n_sens_eng : int           ,
                  n_sens_trk : int           ,
                  length_eng : int           ,
                  length_trk : int           ,
                  group_name : str =     None,
                  compression: str =  'ZLIB4') -> Callable[[int, List, List, List], None]:
    """
    Generalised buffer writer which defines a raw waveform writer
    for each type of sensor as well as an event info writer
    with written event, timestamp and a mapping to the
    nexus event number in case of event splitting.
    """

    eng_writer = rwf_writer(h5out,
                            group_name      =  group_name,
                            compression     = compression,
                            table_name      =     'pmtrd',
                            n_sensors       =  n_sens_eng,
                            waveform_length =  length_eng)

    trk_writer = rwf_writer(h5out,
                            group_name      =  group_name,
                            compression     = compression,
                            table_name      =    'sipmrd',
                            n_sensors       =  n_sens_trk,
                            waveform_length =  length_trk)

    try:
        evt_group = getattr(h5out.root, 'Run')
    except tb.NoSuchNodeError:
        evt_group = h5out.create_group(h5out.root, 'Run')

    evt_tbl   = h5out.create_table(evt_group, "events", EventInfo,
                                   "event & timestamp \
                                    for each index",
                                   tbl.filters(compression))
    nexus_map = h5out.create_table(evt_group, "eventMap", EventMap,
                                   "event & nexus evt \
                                    for each index",
                                   tbl.filters(compression))

    def write_buffers(nexus_evt     :        int ,
                      eng_sens_order: List[  int],
                      trk_sens_order: List[  int],
                      timestamps    : List[  int],
                      events        : List[Tuple]) -> None:

        for t_stamp, (eng, trk) in zip(timestamps, events):
            row  = evt_tbl.row
            row ["event_number"] = write_buffers.counter
            row ["timestamp"]    = t_stamp
            row .append()
            mrow = nexus_map.row
            mrow["event_number"] = write_buffers.counter
            mrow["nexus_evt"]    = nexus_evt
            mrow.append()

            e_sens = np.zeros((n_sens_eng, length_eng), np.int)
            t_sens = np.zeros((n_sens_trk, length_trk), np.int)

            e_sens[eng_sens_order] = eng
            eng_writer(e_sens)

            t_sens[trk_sens_order] = trk
            trk_writer(t_sens)

            write_buffers.counter += 1
    write_buffers.counter = 0
    return write_buffers


def load_sensors(file_names: List[str],
                 db_file   :      str ,
                 run_no    :      int ) -> Generator:
    """
    Loads the nexus MC sensor information into
    a pandas DataFrame using the IC function
    load_mcsensor_response_df.
    Returns info event by event in as a
    generator in the structure expected by
    the dataflow.

    file_names : List of strings
                 List of input file names to be read
    db_file    : string
                 Name of detector database to be used
    run_no     : int
                 Run number for database
    """
    pmt_ids  = DB.DataPMT (db_file, run_no).SensorID
    sipm_ids = DB.DataSiPM(db_file, run_no).SensorID

    for file_name in file_names:

        sns_resp = load_mcsensor_response_df(file_name        ,
                                             db_file = db_file,
                                             run_no  = run_no )
        sns_bins    = get_sensor_binning(file_name)
        pmt_binwid  = sns_bins.bin_width[sns_bins.index.str.contains( 'Pmt')]
        sipm_binwid = sns_bins.bin_width[sns_bins.index.str.contains('SiPM')]

        timestamps = event_timestamp(file_name)

        for evt in sns_resp.index.levels[0]:

            pmt_sig  = sns_resp.loc[evt].index.isin( pmt_ids)
            pmt_wfs  = sns_resp.loc[evt][ pmt_sig]
            sipm_sig = sns_resp.loc[evt].index.isin(sipm_ids)
            sipm_wfs = sns_resp.loc[evt][sipm_sig]

            yield dict(evt         = evt                ,
                       timestamp   = timestamps()       ,
                       pmt_binwid  = pmt_binwid .iloc[0],
                       sipm_binwid = sipm_binwid.iloc[0],
                       pmt_wfs     = pmt_wfs            ,
                       sipm_wfs    = sipm_wfs           )


def load_hits(file_names: List[str]) -> Generator:
    """
    Loads mc hit info into a pandas DataFrame
    using the IC function read_mchits_df.
    Returns this information as well as timestamp
    and general mc info in the generator format
    expected by the dataflow.

    files_names : list of strings
                  List of nexus file names to be read.
    """

    for file_name in file_names:

        hits_df = load_mchits_df(file_name)
        timestamps = event_timestamp(file_name)
        with tb.open_file(file_name) as h5in:

            mc_info    = tbl.get_mc_info(h5in)

            for evt in hits_df.index.levels[0]:
                yield dict(evt       = evt             ,
                           mc        = mc_info         ,
                           timestamp = timestamps()    ,
                           hits      = hits_df.loc[evt])
