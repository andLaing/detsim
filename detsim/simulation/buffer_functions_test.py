import numpy as np

from pytest import fixture
from pytest import    mark

import invisible_cities.database.load_db as DB

from invisible_cities.io  .mcinfo_io         import        get_sensor_binning
from invisible_cities.io  .mcinfo_io         import load_mcsensor_response_df
from invisible_cities.core import system_of_units as      units

from . buffer_functions import         wf_binner
from . buffer_functions import calculate_buffers
from . buffer_functions import     signal_finder


@fixture(scope="module")
def mc_waveforms(fullsim_data):
    wfs         = load_mcsensor_response_df(fullsim_data   ,
                                            db_file = 'new',
                                            run_no  = -6400)
    sns_bins    = get_sensor_binning(fullsim_data)
    pmt_binwid  = sns_bins.bin_width[sns_bins.index.str.contains( 'Pmt')]
    sipm_binwid = sns_bins.bin_width[sns_bins.index.str.contains('SiPM')]
    return wfs.index.levels[0], pmt_binwid.iloc[0], sipm_binwid.iloc[0], wfs


## !! to-do: generalise for all detector configurations
@fixture(scope="module")
def pmt_ids():
    return DB.DataPMT('new', 6400).SensorID.values


@fixture(scope="module")
def sipm_ids():
    return DB.DataSiPM('new', 6400).SensorID.values


@fixture(scope="module")
def binned_waveforms(mc_waveforms, pmt_ids, sipm_ids):
    max_buffer = 10 * units.minute
    wf_binner_  = wf_binner(max_buffer)

    evts, pmt_binwid, sipm_binwid, all_wfs = mc_waveforms

    evt   = evts[0]
    wfs   = all_wfs.loc[evt]

    pmt_indx  = wfs.index.isin(pmt_ids)
    pmts  = wfs[ pmt_indx]
    sipms = wfs[~pmt_indx]

    ## Assumes pmts the triggering sensors as in new/next-100
    bins_min = pmts.time.min()
    bins_max = pmts.time.max() + pmt_binwid
    pmt_bins ,  pmt_wf = wf_binner_(pmts ,  pmt_binwid,
                                    bins_min, bins_max)
    sipm_bins, sipm_wf = wf_binner_(sipms, sipm_binwid,
                                    bins_min, bins_max)
    return pmt_bins, pmt_wf, sipm_bins, sipm_wf


def test_wf_binner(mc_waveforms, pmt_ids, sipm_ids, binned_waveforms):

    pmt_bins, pmt_wf, sipm_bins, sipm_wf = binned_waveforms

    evts, pmt_binwid, sipm_binwid, all_wfs = mc_waveforms

    evt   = evts[0]
    wfs   = all_wfs.loc[evt]

    pmt_indx  = wfs.index.isin(pmt_ids)
    pmts  = wfs[ pmt_indx]
    sipms = wfs[~pmt_indx]

    assert np.all(np.diff( pmt_bins) ==  pmt_binwid)
    assert np.all(np.diff(sipm_bins) == sipm_binwid)
    assert pmt_bins[ 0] >= sipm_bins[ 0]
    assert pmt_bins[-1] >= sipm_bins[-1]

    pmt_sum  = pmts .charge.sum()
    sipm_sum = sipms.charge.sum()
    assert pmt_wf .sum().sum() ==  pmt_sum
    assert sipm_wf.sum().sum() == sipm_sum


@mark.parametrize("signal_thresh", (2, 10))
def test_signal_finder(binned_waveforms, signal_thresh):

    pmt_bins, pmt_wf, *_ = binned_waveforms

    buffer_length = 800
    bin_width     = np.diff(pmt_bins)[0]

    sig_finder    = signal_finder(buffer_length,
                                  bin_width    ,
                                  signal_thresh)

    pmt_sum       = pmt_wf.sum()
    pulses        = sig_finder(pmt_wf)

    assert np.all(pmt_sum[pulses] > signal_thresh)


@mark.parametrize("pre_trigger signal_thresh".split(),
                  ((100,  2),
                   (400, 10)))
def test_calculate_buffers(mc_waveforms, binned_waveforms,
                           pre_trigger ,    signal_thresh):

    _, pmt_binwid, sipm_binwid, _ = mc_waveforms

    pmt_bins, pmt_wf, sipm_bins, sipm_wf = binned_waveforms

    buffer_length     = 800

    sig_finder        = signal_finder(buffer_length,
                                      pmt_binwid   ,
                                      signal_thresh)

    pulses            = sig_finder(pmt_wf)

    buffer_calculator = calculate_buffers(buffer_length,
                                          pre_trigger  ,
                                          pmt_binwid   ,
                                          sipm_binwid  )

    buffers           = buffer_calculator(pulses, *binned_waveforms)
    pmt_sum           = pmt_wf.sum()

    assert len(buffers) == len(pulses)
    for i, evt in enumerate(buffers):
        sipm_trg_bin = np.where(sipm_bins <= pmt_bins[pulses[i]])[0][-1]
        diff_binedge = pmt_bins[pulses[i]] - sipm_bins[sipm_trg_bin]
        pre_trg_samp = int(pre_trigger * units.mus / pmt_binwid + diff_binedge)

        assert pmt_wf .shape[0] == evt[0].shape[0]
        assert sipm_wf.shape[0] == evt[1].shape[0]
        assert evt[0] .shape[1] == int(buffer_length * units.mus / pmt_binwid)
        assert np.sum(evt[0], axis=0)[pre_trg_samp] == pmt_sum[pulses[i]]
