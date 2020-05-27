import os

import numpy  as np
import pandas as pd
import tables as tb

from pytest import fixture
from pytest import    mark

from invisible_cities.core.configure     import              configure
from invisible_cities.core.testing_utils import assert_tables_equality

from . position_signal import position_signal


def test_position_signal_kr(config_tmpdir, fullsim_data, test_config):

    PATH_OUT = os.path.join(config_tmpdir, 'Kr_fullsim.buffers.h5')

    conf = configure(['dummy', test_config])
    conf.update(dict(files_in = fullsim_data,
                     file_out = PATH_OUT    ))


    cnt = position_signal(conf.as_namespace)

    with tb.open_file(fullsim_data, mode='r') as h5in, \
         tb.open_file(PATH_OUT    , mode='r') as h5out:

        assert hasattr(h5out.root   ,        'MC')
        assert hasattr(h5out.root   ,       'Run')
        assert hasattr(h5out.root   ,     'pmtrd')
        assert hasattr(h5out.root   ,    'sipmrd')
        assert hasattr(h5out.root.MC,      'hits')
        assert hasattr(h5out.root.MC, 'particles')

        #assert_tables_equality(h5in .root.MC.particles,
        #                       h5out.root.MC.particles)


def test_position_signal_neut(config_tmpdir, neut_fullsim,
                              test_config  , neut_buffers):

    PATH_OUT = os.path.join(config_tmpdir, 'neut_fullsim.buffers.h5')

    conf = configure(['dummy', test_config])
    conf.update(dict(files_in = neut_fullsim,
                     file_out = PATH_OUT    ))


    cnt = position_signal(conf.as_namespace)

    with tb.open_file(neut_buffers, mode='r') as h5test, \
         tb.open_file(PATH_OUT    , mode='r') as h5out:

         pmt_out  = h5out .root.pmtrd
         pmt_test = h5test.root.pmtrd

         assert pmt_out.shape == pmt_test.shape
         assert_tables_equality(pmt_out, pmt_test)

         sipm_out  = h5out .root.sipmrd
         sipm_test = h5test.root.sipmrd

         assert sipm_out.shape == sipm_test.shape
         assert_tables_equality(sipm_out, sipm_test)
