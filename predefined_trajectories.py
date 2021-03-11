import numpy as np


def circle_trajectory(t, tf, r):
    # circle
    w = 2 * np.pi / tf
    return np.concatenate((r * np.cos(w * t).reshape((-1, 1)), r * np.sin(w * t).reshape((-1, 1))), axis=1)
