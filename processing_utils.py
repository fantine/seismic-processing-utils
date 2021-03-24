import glob
import os
from typing import List, Sequence, Text

import numpy as np
from scipy import signal


def get_filenames(file_pattern: Text) -> List[Text]:
  return sorted(glob.glob(os.path.expanduser(file_pattern)))


def get_strain_rate(data: Sequence[float]) -> Sequence[float]:
  """Computes strain rate across the second dimension."""
  return np.pad(data[:, 1:] - data[:, :-1], ((0, 0), (0, 1)), mode='edge')


def remove_median(data: Sequence[float]) -> Sequence[float]:
  """Removes the median across the first dimension."""
  return data - np.median(data, axis=0, keepdims=True)


def clip(data, clip_percentile):
  min_val = np.percentile(data, clip_percentile)
  max_val = np.percentile(data, 100. - clip_percentile)
  return np.clip(data, min_val, max_val)


def bandpass(data, low, high, dt, axis=-1, order=6):
  nyquist = 0.5 / dt
  low = low / nyquist
  high = high / nyquist
  sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
  return np.float32(signal.sosfiltfilt(sos, data, axis=axis))


def lowpass(data, high, dt, axis=-1, order=5):
  nyquist = 0.5 / dt
  high = high / nyquist
  sos = signal.butter(order, high, btype='lowpass', output='sos')
  return np.float32(signal.sosfiltfilt(sos, data, axis=axis))


def highpass(data, low, dt, axis=-1, order=5):
  nyquist = 0.5 / dt
  low = low / nyquist
  sos = signal.butter(order, low, btype='highpass', output='sos')
  return np.float32(signal.sosfiltfilt(sos, data, axis=axis))


def decimate(data, q, axis=-1):
  return np.float32(signal.decimate(data, q=q, axis=axis))


def normalize(data, axis=-1):
  return data / np.std(data, axis=axis, keepdims=True)
