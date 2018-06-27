import numpy as np


def compute_errors(regression):
    # Compute mean squared error of the regression components
    x = regression[0]
    y = regression[1]
    u = regression[2]
    v = regression[3]
    ur = regression[4]
    vr = regression[5]

    ue_raw = np.subtract(u, ur)
    ve_raw = np.subtract(v, vr)

    ue_scaled = np.divide(np.subtract(u, ur), u)
    ve_scaled = np.divide(np.subtract(v, vr), v)

    ue_av_raw = np.sum(ue_raw) / ue_raw.size
    ve_av_raw = np.sum(ve_raw) / ve_raw.size

    ue_av_scaled = np.sum(ue_scaled) / ue_scaled.size
    ve_av_scaled = np.sum(ve_scaled) / ve_scaled.size

    ge_av_raw = np.add(ue_av_raw, ve_av_raw) / ue_raw.size
    ge_av_scaled = np.add(ue_av_scaled, ve_av_scaled) / ve_raw.size

    return ue_raw, ve_raw, ue_av_raw, ve_av_raw, ge_av_raw, ue_scaled, ve_scaled, ue_av_scaled, ve_av_scaled, ge_av_scaled



