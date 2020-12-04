import argparse
import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile


def get_hparams():
    parser = argparse.ArgumentParser()
    # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To
    # test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    parser.add_argument('--fmin', default=55)
    # To be increased/reduced depending on data.
    parser.add_argument('--fmax', default=7600)
    # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2,
    # faster and cleaner convergence)
    parser.add_argument('--symmetric_mels', default=True)
    # whether to apply filter
    parser.add_argument('--preemphasize', default=True)
    # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    parser.add_argument('--hop_size', default=200)
    # 16000Hz (corresponding to librispeech) (sox --i <filename>)
    parser.add_argument('--sample_rate', default=16000)
    # filter coefficient.
    parser.add_argument('--preemphasis', default=0.97)
    # Mel and Linear spectrograms normalization/scaling and clipping
    parser.add_argument('--signal_normalization', default=True)
    # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not
    # be too big to avoid gradient explosion,
    # not too small for fast convergence)
    # Contribution by @begeekmyfriend
    # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude
    # levels. Also allows for better G&L phase reconstruction)
    parser.add_argument('--max_abs_value', default=4.0)
    # Number of mel-spectrogram channels and local conditioning dimensionality
    parser.add_argument('--num_mels', default=80)
    # Limits
    parser.add_argument('--ref_level_db', default=20)
    # Limits
    parser.add_argument('--min_level_db', default=-100)
    # Extra window size is filled with 0 paddings to match this parameter
    parser.add_argument('--n_fft', default=800)
    # Can replace hop_size parameter. (Recommended: 12.5)
    parser.add_argument('--frame_shift_ms', default=None)
    # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
    # It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
    # Does not work if n_ffit is not multiple of hop_size!!
    parser.add_argument('--use_lws', default=False)
    # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    parser.add_argument('--win_size', default=800)
    # Only relevant if mel_normalization = True
    parser.add_argument('--allow_clipping_in_normalization', default=True)

    args = parser.parse_args()
    return args


hp = get_hparams()


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size


def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db

    if hp.signal_normalization:
        return _normalize(S)
    return S


def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db

    if hp.signal_normalization:
        return _normalize(S)
    return S


def _lws_processor():
    import lws
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode='speech')


def _stft(y):
    if hp.use_lws:
        return _lws_processor(hp).stft(y).T
    else:
        return librosa.stft(y=y,
                            n_fft=hp.n_fft,
                            hop_length=get_hop_size(),
                            win_length=hp.win_size)


##########################################################
#Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    '''Compute number of time frames of spectrogram'''
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    '''Compute left and right padding'''
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]


# Conversions
_mel_basis = None


def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(hp.sample_rate,
                               hp.n_fft,
                               n_mels=hp.num_mels,
                               fmin=hp.fmin,
                               fmax=hp.fmax)


def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _normalize(S):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip(
                (2 * hp.max_abs_value) *
                ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                -hp.max_abs_value, hp.max_abs_value)
        else:
            return np.clip(
                hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)),
                0, hp.max_abs_value)

    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) /
                                         (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))


def _denormalize(D):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (((np.clip(D, -hp.max_abs_value, hp.max_abs_value) +
                      hp.max_abs_value) * -hp.min_level_db /
                     (2 * hp.max_abs_value)) + hp.min_level_db)
        else:
            return ((np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db /
                     hp.max_abs_value) + hp.min_level_db)

    if hp.symmetric_mels:
        return (((D + hp.max_abs_value) * -hp.min_level_db /
                 (2 * hp.max_abs_value)) + hp.min_level_db)
    else:
        return ((D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
