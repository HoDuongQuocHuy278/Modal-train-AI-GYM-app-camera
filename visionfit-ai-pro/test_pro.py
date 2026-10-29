"""Pro-specific offline smoke tests; all use synthetic data, not accuracy benchmarks."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from build_dataset import build
from live import ActionSmoother
from local_video import body_visible, open_local_capture
from metrics import classification_report
from pose_model import PoseNet, decode_pose, pose_loss, infer_rgb, IMAGE_SIZE
from repetition import PushUpCounter, LungeCounter
from visionfit import split_train_val_test


def test_pose_model_forward_and_decoder():
    torch.set_num_threads(2)
    model = PoseNet().eval()
    with torch.no_grad():
        heat, visible = model(torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert heat.shape == (2, 33, 32, 32)
    assert visible.shape == (2, 33)
    decoded = decode_pose(heat, visible)
    assert decoded.shape == (2, 33, 4)
    assert torch.all(decoded[..., :2] >= 0) and torch.all(decoded[..., :2] <= 1)
    assert torch.isfinite(pose_loss(heat, visible, torch.ones(2, 33, 3) * .5))
    kp = infer_rgb(model, np.zeros((128, 128, 3), dtype='uint8'))
    assert kp.shape == (33, 4) and np.allclose(kp[:, 2], 0)


def test_train_val_test_subject_disjoint():
    groups = np.repeat([f'p{i}' for i in range(10)], 8)
    labels = np.tile(np.arange(4), 20)
    train, val, test = split_train_val_test(labels, groups, 1)
    assert not set(groups[train]) & set(groups[val])
    assert not set(groups[train]) & set(groups[test])
    assert not set(groups[val]) & set(groups[test])
    assert len(train) + len(val) + len(test) == len(labels)


def test_metrics_known_case():
    m = classification_report([0, 0, 1, 1], [0, 1, 1, 1], ['squat', 'rest'])
    assert m['confusion_matrix'] == [[1, 1], [0, 2]]
    assert m['accuracy'] == .75
    assert m['per_class']['squat']['recall'] == .5


def test_smoother_unknown_and_stable():
    s = ActionSmoother(['squat', 'rest'], alpha=1, threshold=.65, min_votes=2)
    assert s.update([.8, .2])[0] == 'unknown'
    assert s.update([.8, .2])[0] == 'squat'
    assert s.update([.5, .5])[0] == 'squat'
    assert s.update([.5, .5])[0] == 'unknown'
    s.clear()
    assert s.current == 'unknown'


def test_pushup_and_lunge_reps():
    frame = np.zeros((33, 4), dtype='float32')
    frame[:, 3] = 1
    for shoulder, elbow, wrist in [(11, 13, 15), (12, 14, 16)]:
        frame[shoulder, :2] = [0, 0]
        frame[elbow, :2] = [0, 1]
        frame[wrist, :2] = [0, 2]
    c = PushUpCounter()
    assert c.update(frame)['reps'] == 0
    frame[[11, 12], :2] = [1, 1]
    assert c.update(frame)['phase'] == 'down'
    frame[[11, 12], :2] = [0, 0]
    assert c.update(frame)['reps'] == 1
    c.reset_phase()
    assert c.reps == 1
    l = LungeCounter()
    for hip, knee, ankle in [(23, 25, 27), (24, 26, 28)]:
        frame[hip, :2] = [0, 0]; frame[knee, :2] = [0, 1]; frame[ankle, :2] = [0, 2]
    l.update(frame)
    frame[23, :2] = [1, 1]
    assert l.update(frame)['phase'] == 'down'
    frame[23, :2] = [0, 0]
    assert l.update(frame)['reps'] == 1


def test_builder_and_quality_filter(tmp_path):
    labels = ['squat', 'standing']
    (tmp_path / 'labels.json').write_text(json.dumps(labels))
    for action in labels:
        for person in range(5):
            folder = tmp_path / 'clips' / action / f's{person}'
            folder.mkdir(parents=True)
            arr = np.zeros((12, 33, 4), dtype='float32')
            arr[:, :, 3] = 1
            np.save(folder / 'take.npy', arr)
    dest = build(tmp_path / 'clips', tmp_path / 'labels.json', tmp_path / 'out.npz')
    with np.load(dest, allow_pickle=False) as d:
        assert d['X'].shape == (10, 32, 33, 4)
        assert len(np.unique(d['subjects'])) == 5
    frame = np.zeros((33, 4), dtype='float32')
    assert not body_visible(frame)
    frame[[11, 12, 23], 3] = 0.8
    assert body_visible(frame)


def test_reject_network_source():
    with pytest.raises(ValueError, match='network'):
        open_local_capture('https://example.com/video.mp4')
