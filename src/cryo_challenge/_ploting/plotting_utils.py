import numpy as np

def res_at_fsc_threshold(fscs, threshold=0.5):
    res_fsc_half = np.argmin(fscs > threshold, axis=-1)
    fraction_nyquist = 0.5*res_fsc_half / fscs.shape[-1]
    return res_fsc_half, fraction_nyquist