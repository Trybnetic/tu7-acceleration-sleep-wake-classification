import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from numpy.lib.stride_tricks import sliding_window_view


# pylint: disable=C0103
def cole_kripke(data, weights=[106, 54, 58, 76, 230, 74, 67], P=0.001):
    """
    Implementation of the Algorithm proposed by Cole et al. (1992).

    See
    Cole, R. J., Kripke, D. F., Gruen, W., Mullaney, D. J., & Gillin, J. C. (1992). Automatic
    sleep/wake identification from wrist activity. Sleep, 15(5), 461-469.

    Parameters
    ----------
    data: np.array
        a numpy array containing the the recorded data of one patient over time.
    weights: tuple, list or np.array
        a 7-tuple of weights for the entries of the corresponding window. The
        default values correspond to the optimal parameters proposed for one
        minute epochs in Cole et al. (1992)
    P: float
        a scale factor for the entire equation

    Returns
    -------
    sleep_data : np.array
        a boolean numpy array stating whether the epoch is considered to be asleep

    """

    # Actigraph treats missing epochs as 0, see https://actigraphcorp.force.com/support/
    # s/article/Where-can-I-find-documentation-for-the-Sadeh-and-Cole-Kripke-algorithms
    data = np.concatenate([[0] * 4, data, [0] * 2])

    sliding_windows = np.stack([data[xx:xx-6] for xx in range(6)], axis=-1)

    def _score(window, weights=weights, P=P):
        return P * sum([xx * yy for xx, yy in zip(window, weights)])

    sleep_data = np.apply_along_axis(_score, 1, sliding_windows)

    return sleep_data < 1


def sadeh(data, weights=[7.601, 0.065, 1.08, 0.056, 0.703]):
    """
    Implementation of the Algorithm proposed by Sadeh et al. (1994).

    See
    Sadeh, A., Sharkey, M., & Carskadon, M. A. (1994). Activity-based sleep-wake
    identification: an empirical test of methodological issues. Sleep, 17(3), 201-207.

    Parameters
    ----------
    data: np.array
        a numpy array containing the the recorded data of one patient over time.
    weights: tuple, list or np.array
        a 5-tuple of weights for the different parameters. The default values
        correspond to the optimal parameters proposed in Sadeh et al. (1994)
    P: float
        a scale factor for the entire equation

    Returns
    -------
    sleep_data : np.array
        a boolean numpy array stating whether the epoch is considered to be asleep

    """

    # Actigraph treats missing epochs as 0, see https://actigraphcorp.force.com/support/
    # s/article/Where-can-I-find-documentation-for-the-Sadeh-and-Cole-Kripke-algorithms
    data = np.concatenate([[0] * 5, data, [0] * 6])

    sliding_windows = np.stack([data[xx:xx-11] for xx in range(11)], axis=-1)

    def _score(window, weights=weights):

        if window[5] == 0:
            log_res = 0  # use 0 if epoch count is zero
        else:
            log_res = np.log(window[5])

        return weights[0] - (weights[1] * np.average(window)) - (weights[2] * np.sum((window >= 50) & (window < 100))) - (weights[3] * np.std(window)) - (weights[4] * log_res)

    sleep_data = np.apply_along_axis(_score, 1, sliding_windows)

    return sleep_data > -4


def calc_metrics(seq_true, seq_pred):
    accuracy = metrics.accuracy_score(seq_true, seq_pred)
    precision = metrics.precision_score(seq_true, seq_pred)
    recall = metrics.recall_score(seq_true, seq_pred)
    f1_score = metrics.f1_score(seq_true, seq_pred)

    return accuracy, precision, recall, f1_score


def load_data(file_path, subject_id, target_dict, resampled_frequency="1s", colnames=["Time", "X", "Y", "Z", "Sleep Stage"]):
    data = pd.read_hdf(file_path, key=subject_id)
    time_col, x_col, y_col, z_col, target_col = colnames

    if resampled_frequency:
        tmp = data.set_index(time_col).resample(resampled_frequency).mean().reset_index()
        tmp = pd.merge_asof(tmp[[time_col, x_col, y_col, z_col]], data[[time_col, target_col]], on=time_col, direction='nearest')
        tmp = tmp.loc[~tmp[target_col].isnull(), :]
    else:
        tmp = data

    X = tmp[[x_col, y_col, z_col]].values

    X = torch.from_numpy(X)
    y = torch.from_numpy(tmp[target_col].apply(lambda label: target_dict[label]).values)

    assert X.shape[0] == y.shape[0]

    return X, y


def hz_to_mel(hz):
    return (2595 * np.log10(1 + (hz / 2) / 700))


def mel_to_hz(mel):
    return (700 * (10**(mel / 2595) - 1))


def fbanks(signal, sample_rate, win_len, win_step, NFFT=512, nfilt=40):
    """
    see https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    """
    frame_len, frame_step = int(win_len * sample_rate), int(win_step * sample_rate)

    frames = sliding_window_view(signal, frame_len)[::frame_step].copy()
    frames *= np.hamming(frame_len)

    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2)

    low_freq_mel = 0
    high_freq_mel = hz_to_mel(sample_rate)

    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = mel_to_hz(mel_points)

    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    # remove zeros for log
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB

    return hz_points, filter_banks



def load_mel_data(file_path, subject_id, target_dict, win_len=60, win_step=60, sample_rate=85.7, resampled_frequency="1min", colnames=["Time", "X", "Y", "Z", "Sleep Stage"]):
    data = pd.read_hdf(file_path, key=subject_id)
    time_col, x_col, y_col, z_col, target_col = colnames

    emno = np.sqrt(np.sum(np.square(data[[x_col, y_col, z_col]]), axis=1)) - 1

    _, fbanks_x = fbanks(data[x_col], sample_rate, win_len, win_step)
    _, fbanks_y = fbanks(data[y_col], sample_rate, win_len, win_step)
    _, fbanks_z = fbanks(data[z_col], sample_rate, win_len, win_step)
    _, fbanks_emno = fbanks(emno, sample_rate, win_len, win_step)

    X = np.hstack([fbanks_x, fbanks_y, fbanks_z, fbanks_emno])

    tmp = data.set_index(time_col)[target_col].resample(resampled_frequency).pad()
    tmp = tmp[~tmp.isnull()]
    y = tmp.apply(lambda label: target_dict[label]).values

    y = y[:X.shape[0]] # Sometimes due to rounding errors, we get one elem more in
                       # y than in X (likely caused by the sliding_window_view). In
                       # these cases, we ignore the additional y value

    X, y = torch.from_numpy(X), torch.from_numpy(y)

    assert X.shape[0] == y.shape[0]

    return X, y


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def init_weights2(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
