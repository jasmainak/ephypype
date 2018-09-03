import os.path as op

import mne
from mne.utils import _TempDir

from ephypype.power import compute_and_save_psd


data_path = mne.datasets.sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=False)
raw.crop(0, 60)


def test_power():
    """Test computing and saving PSD."""
    fmin = 0.1
    fmax = 300
    event_id = None
    tmin, tmax = -0.2, 0.5
    events = mne.find_events(raw)
    tempdir = _TempDir()
    epochs_fname = op.join

    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False,
                           eog=True, exclude=[])
    # raise error if preload is false
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        picks=picks, baseline=(None, 0),
                        reject=None, preload=False)
    tempdir = _TempDir()
    epochs_fname = op.join(tempdir, 'test-epo.fif')
    epochs.save(epochs_fname)

    compute_and_save_psd(epochs_fname, fmin, fmax, method='welch')
    compute_and_save_psd(epochs_fname, fmin, fmax, method='multitaper')
