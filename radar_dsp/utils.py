# Copyright 2019 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
try:
    from enum import Enum
except ImportError:
    print("enum only exists in Python 3.4 or newer")

try:
    class Window(Enum):
        BARTLETT = 1
        BLACKMAN = 2
        HAMMING  = 3
        HANNING  = 4
except NameError:
    class Window:
        BARTLETT = 1
        BLACKMAN = 2
        HAMMING  = 3
        HANNING  = 4

RANGEIDX = 0
DOPPLERIDX = 1
PEAKVAL = 2

MAX_OBJ_OUT = 100

def windowing(input, window_type, axis=0):
    """Window the input based on given window type.

    Args:
        input: input numpy array to be windowed.

        window_type: enum chosen between Bartlett, Blackman, Hamming, Hanning and Kaiser.

        axis: the axis along which the windowing will be applied.
    
    Returns:

    """

    # 输入为(num_chirps_per_frame, num_adc_samples, num_rx_antennas)
    # 要转(num_chirps_per_frame, num_rx_antennas, num_adc_samples)

    axis = np.core.numeric.normalize_axis_index(axis, input.ndim)
    window_length = input.shape[axis]
    if window_type == Window.BARTLETT:
        window = np.bartlett(window_length)
    elif window_type == Window.BLACKMAN:
        window = np.blackman(window_length)
    elif window_type == Window.HAMMING:
        window = np.hamming(window_length)
    elif window_type == Window.HANNING:
        window = np.hanning(window_length)
    else:
        raise ValueError("The specified window is not supported!!!")
    # Reshape the 1-D coefficients so they broadcast along the requested
    # dimension, including when that dimension is not the last one.
    window_shape = [1] * input.ndim
    window_shape[axis] = window_length
    output = input * window.reshape(window_shape)
    return output
