import numpy as np


def generate_simple(w, h, max_permeability=2, min_depth=1, max_depth=5, max_height=2):
    centers = [(np.random.randint(0, w), np.random.randint(0, h)) for _ in range(np.random.randint(2, 5))]
    widths = (w**2 + h**2) * (np.random.rand(1, 1, len(centers)) + 0.1) / len(centers)**2

    map = np.zeros((h, w, 4))
    # 0 - permeability
    # 1 - depth
    # 2 - height
    # 3 - porosity

    column = np.linspace(0, w-1, w).reshape((1, -1, 1)).repeat(h, axis=0).repeat(len(centers), axis=-1)
    row = np.linspace(0, h-1, h).reshape((-1, 1, 1)).repeat(w, axis=1).repeat(len(centers), axis=-1)
    dst = np.stack([column, row], axis=-1)
    ctr = np.array(centers).reshape((1, 1, -1, 2)).repeat(h, axis=0).repeat(w, axis=1)
    dst = ((ctr - dst)**2).sum(-1)
    dst = dst / widths
    decay = np.e ** (-np.clip(dst - 0.25, 0., None))
    decay = np.max(decay, axis=-1)

    map[:, :, 3] = decay * (0.2 * np.random.rand(*decay.shape) + 0.8)
    map[:, :, 2] = decay * max_height
    map[:, :, 1] = min_depth + (max_depth - min_depth) * decay / 2

    if np.random.rand() < 0.44:
        decay = np.flip(decay, axis=0)

    if np.random.rand() < 0.49:
        decay = np.flip(decay, axis=1)

    decay = np.roll(decay, (np.random.randint(0, h), np.random.randint(0, w)), axis=(0, 1))
    map[:, :, 0] = decay * max_permeability + 0.1

    return map