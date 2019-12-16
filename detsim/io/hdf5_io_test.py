import os

import numpy  as np
import pandas as pd
import tables as tb

from numpy.random import randint

from functools    import partial

from pytest import  approx
from pytest import fixture
from pytest import    mark
from pytest import  raises

from invisible_cities.io  .mcinfo_io         import load_mcsensor_response_df
from invisible_cities.io  .mcinfo_io         import        get_sensor_binning
from invisible_cities.core.system_of_units_c import                     units

from . hdf5_io import   buffer_writer
from . hdf5_io import event_timestamp
from . hdf5_io import       load_hits
from . hdf5_io import    load_sensors
from . hdf5_io import   save_run_info

from ..simulation.buffer_functions import calculate_buffers
from ..util      .util             import     trigger_times


def test_save_run_info(config_tmpdir):

    run_number = -6400

    out_name = os.path.join(config_tmpdir, 'test_runInfo.h5')
    with tb.open_file(out_name, 'w') as h5out:

        save_run_info(h5out, run_number)

    with tb.open_file(out_name) as h5saved:

        assert 'Run'     in h5saved.root
        assert 'runInfo' in h5saved.root.Run
        assert h5saved.root.Run.runInfo[0][0] == run_number


def test_event_timestamp(fullsim_data):

    with tb.open_file(fullsim_data) as data_in:

        n_evt = len(data_in.root.MC.events)
        ## Explicitly extract timestamps
        first_hit = 0
        timestamps = []
        for ext in data_in.root.MC.extents:
            timestamps.append(data_in.root.MC.hits[first_hit][2])
            first_hit = ext[2] + 1

        time_stamp = event_timestamp(data_in)
        for i in range(n_evt):
            assert time_stamp() == approx(timestamps[i])
        with raises(IndexError):
            time_stamp()


@fixture(scope = 'module')
def event_definitions(fullsim_data):
    len_energy    = 100
    len_tracking  =  10

    buffer_length =  10
    pre_trigger   =   5

    pmt_binwid, sipm_binwid = get_sensor_binning(fullsim_data)

    calculate_buffers_ = calculate_buffers(buffer_length, pre_trigger,
                                           pmt_binwid   , sipm_binwid)

    pmt_bins  = np.arange(300, 13000000,  100)
    sipm_bins = np.arange(0  , 13000000, 1000)

    return len_energy, len_tracking, pmt_bins, sipm_bins, calculate_buffers_


@mark.parametrize("event triggers".split(),
                  ((2, [10]), (3, [10, 1100]), (4, [20])))
def test_buffer_writer(config_tmpdir, event_definitions,
                       event, triggers):
    len_eng, len_trk, pmt_bins, sipm_bins, calc_buffers = event_definitions

    pmt_orders  = list(pd.unique(randint(0,   12, randint(1,  12))))
    sipm_orders = list(pd.unique(randint(0, 1792, randint(1, 500))))

    evt_times = trigger_times(triggers, 0, pmt_bins)

    pmtwf_  = np.random.poisson(5, (len( pmt_orders),  pmt_bins.shape[0]))
    pmtwf   = pd.Series(pmtwf_ .tolist(),  pmt_orders)
    sipmwf_ = np.random.poisson(5, (len(sipm_orders), sipm_bins.shape[0]))
    sipmwf  = pd.Series(sipmwf_.tolist(), sipm_orders)

    buffers = calc_buffers(triggers, pmt_bins, pmtwf, sipm_bins, sipmwf)

    out_name = os.path.join(config_tmpdir, 'test_buffers.h5')
    with tb.open_file(out_name, 'w') as data_out:

        buffer_writer_ = buffer_writer(data_out,
                                       length_eng = len_eng,
                                       length_trk = len_trk)

        buffer_writer_(event, pmt_orders, sipm_orders, evt_times, buffers)


    ## To allow charge comparison
    sipm_bin    = [int(np.where(sipm_bins <= pmt_bins[trg])[0][-1])
                   for trg in triggers]
    sipm_presamp = int(5 * units.mus / 1000)
    pmt_presamp  = [int((pmt_bins[trg] - sipm_bins[sibin]) / 100 + 5 * units.mus / 100) for trg, sibin in zip(triggers, sipm_bin)]
    sipm_possamp = len_trk - sipm_presamp
    pmt_possamp  = [len_eng -  presamp for presamp in pmt_presamp]
    with tb.open_file(out_name) as data_in:

        assert 'Run'    in data_in.root
        assert 'events' in data_in.root.Run
        assert len(data_in.root.Run.events[:]) == len(triggers)

        assert 'detsim' in data_in.root
        assert 'pmtrd'  in data_in.root.detsim
        assert 'sipmrd' in data_in.root.detsim

        pmts  = data_in.root.detsim.pmtrd
        sipms = data_in.root.detsim.sipmrd
        assert len( pmts) == len(triggers)
        assert len(sipms) == len(triggers)

        for i, (trg, pre, pos) in enumerate(zip(triggers   ,
                                                pmt_presamp,
                                                pmt_possamp)):
            file_pmt_sum = np.sum(np.array(pmts[i])[pmt_orders], axis=1)
            slice_       = slice(max(0, trg - pre),
                                 min(pmtwf_.shape[1] - 1, trg + pos))
            assert np.all(file_pmt_sum == np.sum(pmtwf_[:, slice_], axis=1))

        for i, sibin in enumerate(sipm_bin):

            file_sipm_sum = np.sum(np.array(sipms[i])[sipm_orders], axis=1)
            slice_        = slice(max(0, sibin - sipm_presamp),
                                  min(sipmwf_.shape[1] -1, sibin + sipm_possamp))
            assert np.all(file_sipm_sum == np.sum(sipmwf_[:, slice_], axis=1))


def test_load_sensors(fullsim_data):

    #Get basic info about the file
    with tb.open_file(fullsim_data) as h5info:
        n_evt  = len(h5info.root.MC.extents)
        wf_end = [ext[1] for ext in h5info.root.MC.extents]
        n_wfs  = np.diff([0] + wf_end) + 1

    expected_keys = ['evt', 'mc', 'timestamp', 'pmt_binwid',
                     'sipm_binwid',  'pmt_wfs', 'sipm_wfs']

    source  = partial(load_sensors, db_file = 'new', run_no = -6400)
    evt_gen = source((fullsim_data,))

    for i in range(n_evt):
        evt_dict = next(evt_gen)

        assert len(evt_dict) == len(expected_keys)

        data_keys = evt_dict.keys()
        for exp_key in expected_keys:
            assert exp_key in data_keys

        data_nwfs =   evt_dict[ 'pmt_wfs'].shape[0] \
                    + evt_dict['sipm_wfs'].shape[0]
        assert data_nwfs == n_wfs[i]


def test_load_hits(fullsim_data):

    #Get basic info about the file
    with tb.open_file(fullsim_data) as h5info:
        n_evt   = len(h5info.root.MC.extents)
        hit_end = [ext[2] for ext in h5info.root.MC.extents]
        n_hits  = np.diff([0] + hit_end) + 1

    expected_keys = ['evt', 'mc', 'timestamp', 'hits']

    evt_gen = load_hits((fullsim_data,))

    for i in range(n_evt):
        evt_dict = next(evt_gen)

        assert len(evt_dict) == len(expected_keys)

        data_keys = evt_dict.keys()
        for exp_key in expected_keys:
            assert exp_key in data_keys

        assert evt_dict['hits'].shape[0] == n_hits[i]
