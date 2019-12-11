import numpy  as np
import pandas as pd

from pytest import fixture
from pytest import    mark

from .util import first_and_last_times
from .util import         sensor_order


def test_first_and_last_times():

    bin_min    = 22
    bin_max    = 40
    test_bins  = np.arange(bin_min, bin_max)

    bmin, bmax = first_and_last_times(test_bins)

    assert np.round(bmin) == bin_min
    assert np.round(bmax) == bin_max


@fixture(scope = 'function')
def ids_and_orders():
    id_dict = {'new'    :{'pmt_ids' :    (2, 5, 7), 'pmt_ord' :  (2, 5, 7),
                          'sipm_ids': (1010, 5023), 'sipm_ord': (10, 279)},
               'next100':{'pmt_ids' :     (22, 50), 'pmt_ord' :   (22, 50),
                          'sipm_ids': (5044, 7001), 'sipm_ord': (300, 385)}}
    return id_dict
@mark.parametrize("detector", ('new', 'next100'))
def test_sensor_order(ids_and_orders, detector):

    npmt     = len(ids_and_orders[detector]['pmt_ids'])
    pmt_sig  = pd.Series([np.random.uniform(size = 3) for i in range(npmt)],
                         index = ids_and_orders[detector]['pmt_ids'])
    nsipm    = len(ids_and_orders[detector]['sipm_ids'])
    sipm_sig = pd.Series([np.random.uniform(size = 3) for i in range(nsipm)],
                         index = ids_and_orders[detector]['sipm_ids'])

    pmt_ord, sipm_ord = sensor_order(pmt_sig, sipm_sig, detector, -1000)

    assert len( pmt_ord) == npmt
    assert len(sipm_ord) == nsipm

    assert np.all(pmt_ord  == ids_and_orders[detector][ 'pmt_ord'])
    assert np.all(sipm_ord == ids_and_orders[detector]['sipm_ord'])
