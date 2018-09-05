import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

import mne

from ephypype.power import compute_and_save_psd

data_path = mne.datasets.sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'


def test_power():
    """Test computing and saving PSD."""
    fmin = 0.1
    fmax = 300

    compute_and_save_psd(raw_fname, fmin, fmax, method='welch')
    compute_and_save_psd(raw_fname, fmin, fmax, method='multitaper')
