import numpy as np
import torch
from visionfit import ActionNet, SquatCounter, angle, normalize, resample, split_data


def fake_frames(t=12):
    a = np.zeros((t, 33, 4), dtype='float32')
    a[:, :, 3] = 1
    a[:, 11, :2] = [.4, .3]
    a[:, 12, :2] = [.6, .3]
    a[:, 23, :2] = [.45, .5]
    a[:, 24, :2] = [.55, .5]
    return a


def test_dimensions():
    x = normalize(resample(fake_frames(), 32))
    assert x.shape == (32, 33, 4)
    assert np.isfinite(x).all()
    assert ActionNet(4)(torch.from_numpy(x[None])).shape == (1, 4)


def test_angle_and_counter():
    assert abs(angle([1, 0], [0, 0], [0, 1]) - 90) < 1e-5
    c = SquatCounter()
    p = fake_frames(2)[0]
    for hip, knee, ankle in [(23, 25, 27), (24, 26, 28)]:
        p[hip, :2], p[knee, :2], p[ankle, :2] = [0, 0], [0, 1], [0, 2]
    assert c.update(p)['reps'] == 0
    for hip, knee, ankle in [(23, 25, 27), (24, 26, 28)]:
        p[hip, :2] = [1, 1]
    assert c.update(p)['state'] == 'down'
    for hip, knee, ankle in [(23, 25, 27), (24, 26, 28)]:
        p[hip, :2] = [0, 0]
    assert c.update(p)['reps'] == 1


def test_disjoint_subjects():
    groups = np.repeat(np.arange(5), 8).astype(str)
    y = np.tile(np.arange(4), 10)
    a, b = split_data(y, groups, 42)
    assert not set(groups[a]) & set(groups[b])
