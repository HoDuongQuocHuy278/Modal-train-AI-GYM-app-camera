"""VisionFit local skeleton action classifier. No HTTP/API calls."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

from metrics import classification_report

JOINTS = 33
FEATURES = 4
LABELS = ['squat', 'push_up', 'lunge', 'standing']
LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER = 23, 24, 11, 12


def normalize(sequence):
    """Input (T,33,4): normalized image x/y, relative z, visibility in [0,1]."""
    x = np.asarray(sequence, dtype=np.float32).copy()
    if x.ndim != 3 or x.shape[1:] != (JOINTS, FEATURES):
        raise ValueError(f'Expected (T,33,4), got {x.shape}')
    if not np.isfinite(x).all():
        raise ValueError('Coordinates must be finite')
    if x.shape[0] < 2:
        raise ValueError('Need at least two frames')
    x[:, :, 3] = np.clip(x[:, :, 3], 0.0, 1.0)
    # Translation invariance using hip center, scale from shoulder width or torso length.
    hip = (x[:, LEFT_HIP, :3] + x[:, RIGHT_HIP, :3]) / 2
    shoulders = (x[:, LEFT_SHOULDER, :3] + x[:, RIGHT_SHOULDER, :3]) / 2
    shoulder_width = np.linalg.norm(x[:, LEFT_SHOULDER, :2] - x[:, RIGHT_SHOULDER, :2], axis=-1)
    torso = np.linalg.norm((shoulders - hip)[:, :2], axis=-1)
    scale = np.maximum(np.maximum(shoulder_width, torso), 0.05)
    x[:, :, :3] = np.clip((x[:, :, :3] - hip[:, None, :]) / scale[:, None, None], -8, 8)
    # Missing/low confidence landmarks become zero instead of introducing spurious motion.
    x[:, :, :3] *= (x[:, :, 3:4] >= 0.35)
    return x


def resample(seq, length=32):
    """Linear interpolation along time to fixed length."""
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 3 or seq.shape[1:] != (33, 4) or len(seq) < 2:
        raise ValueError('Expected >=2 frames, shape (T,33,4)')
    old = np.linspace(0, 1, len(seq))
    new = np.linspace(0, 1, length)
    out = np.stack([np.interp(new, old, seq[:, j, k]) for j in range(33) for k in range(4)], axis=1)
    return out.reshape(length, 33, 4).astype(np.float32)


class ActionNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(132, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.25), nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # x: (B,T,33,4) -> (B,132,T)
        return self.encoder(x.flatten(2).transpose(1, 2))


def load_data(path, seq_len):
    with np.load(path, allow_pickle=False) as data:
        if not {'X', 'y'}.issubset(data.files):
            raise ValueError('Dataset .npz must have X and y')
        X = data['X'].astype(np.float32)
        y = data['y'].astype(np.int64)
        groups = data['subjects'].astype(str) if 'subjects' in data.files else None
    if X.ndim != 4 or X.shape[2:] != (33, 4) or len(X) != len(y) or len(X) < 8:
        raise ValueError('Expected X (N,T,33,4), y (N,), N>=8')
    if len(np.unique(y)) < 2 or min(y) < 0:
        raise ValueError('At least 2 classes; integer labels >=0')
    if groups is not None and len(groups) != len(y):
        raise ValueError('subjects length must match X')
    return np.stack([normalize(resample(seq, seq_len)) for seq in X]), y, groups


def split_data(y, groups, seed):
    rng = np.random.default_rng(seed)
    if groups is not None:
        unique = np.unique(groups)
        if len(unique) < 3:
            raise ValueError('Need >=3 subjects for subject-disjoint split')
        rng.shuffle(unique)
        n_val = max(1, round(len(unique) * 0.2))
        val_mask = np.isin(groups, unique[:n_val])
        train_ids, val_ids = np.where(~val_mask)[0], np.where(val_mask)[0]
    else:
        print('WARNING: no subjects field; random window split risks data leakage. Record subject IDs.')
        ids = rng.permutation(len(y))
        cut = max(1, round(0.2 * len(y)))
        train_ids, val_ids = ids[cut:], ids[:cut]
    if len(np.unique(y[train_ids])) < len(np.unique(y)):
        raise ValueError('Some classes missing from training set; gather more subjects/clips')
    if not len(val_ids):
        raise ValueError('Empty validation set')
    return train_ids, val_ids


def split_train_val_test(y, groups, seed=42, val_fraction=0.2, test_fraction=0.2):
    """Disjoint subject split. Without IDs, warn that random clip split can leak."""
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError('val_fraction and test_fraction must be positive, sum < 1')
    rng = np.random.default_rng(seed)
    if groups is not None:
        unique = np.unique(groups)
        if len(unique) < 5:
            raise ValueError('Need >=5 different subjects for train/val/test disjoint split')
        units = unique
    else:
        print('WARNING: subject IDs missing: clip split may leak people/videos. Do not report benchmark results.')
        units = np.arange(len(y))
    n_val = max(1, round(len(units) * val_fraction))
    n_test = max(1, round(len(units) * test_fraction))
    if n_val + n_test >= len(units):
        raise ValueError('Not enough samples/subjects for three splits')
    all_classes = set(np.unique(y).tolist())
    for _ in range(200):
        perm = rng.permutation(units)
        val_units = perm[:n_val]
        test_units = perm[n_val:n_val + n_test]
        train_units = perm[n_val + n_test:]
        if groups is not None:
            a = np.where(np.isin(groups, train_units))[0]
            b = np.where(np.isin(groups, val_units))[0]
            c = np.where(np.isin(groups, test_units))[0]
        else:
            a, b, c = train_units, val_units, test_units
        if all_classes.issubset(set(y[a])) and len(b) and len(c):
            return a, b, c
    raise ValueError('Cannot place all classes in train with subject-disjoint splits. Collect more subjects per class.')


class TrainingDataset(Dataset):
    def __init__(self, x, y, augment=False):
        self.x, self.y, self.augment = x, y, augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i].copy()
        if self.augment:
            # Noise only on reliable joints; visibility remains unmodified.
            valid = (x[:, :, 3:4] >= 0.35).astype(np.float32)
            x[:, :, :3] += np.random.normal(0, 0.012, x[:, :, :3].shape).astype(np.float32) * valid
            if np.random.random() < 0.20:
                x[:, :, :3] *= np.float32(np.random.uniform(0.95, 1.05))
        return torch.from_numpy(x), torch.tensor(int(self.y[i]), dtype=torch.long)


def evaluate(model, loader, labels, device):
    model.eval()
    truths, predictions, loss_total, count = [], [], 0.0, 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device))
            loss_total += float(nn.functional.cross_entropy(logits, batch_y.to(device), reduction='sum').item())
            truths.extend(batch_y.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            count += len(batch_y)
    report = classification_report(truths, predictions, labels)
    report['loss'] = round(loss_total / count, 5) if count else None
    return report


def train(args):
    if args.epochs < 1 or args.batch < 1 or args.frames < 2 or args.patience < 1:
        raise ValueError('epochs/batch/patience must >=1; frames >=2')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    X, y, subjects = load_data(args.data, args.frames)
    labels = json.loads(Path(args.labels).read_text(encoding='utf-8')) if args.labels else LABELS
    if not isinstance(labels, list) or len(set(labels)) != len(labels) or any(not isinstance(v, str) or not v for v in labels):
        raise ValueError('Labels must be distinct nonempty strings in a JSON array')
    if sorted(np.unique(y).tolist()) != list(range(len(labels))):
        raise ValueError('Dataset label IDs must match 0..number_of_classes-1')
    train_ids, val_ids, test_ids = split_train_val_test(y, subjects, args.seed, args.val_fraction, args.test_fraction)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but not available')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device))
    model = ActionNet(len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    counts = np.bincount(y[train_ids], minlength=len(labels))
    weights = len(train_ids) / (len(labels) * np.maximum(counts, 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device)) if args.class_weights else nn.CrossEntropyLoss()
    train_dl = DataLoader(TrainingDataset(X[train_ids], y[train_ids], augment=args.augment), batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.from_numpy(X[val_ids]), torch.from_numpy(y[val_ids])), batch_size=args.batch)
    test_dl = DataLoader(TensorDataset(torch.from_numpy(X[test_ids]), torch.from_numpy(y[test_ids])), batch_size=args.batch)
    best, stale = -1.0, 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, seen = 0.0, 0
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(by)
            seen += len(by)
        val_report = evaluate(model, val_dl, labels, device)
        row = {'epoch': epoch, 'train_loss': round(loss_sum / seen, 5),
               'val_loss': val_report['loss'], 'val_accuracy': val_report['accuracy'],
               'val_macro_f1': val_report['macro_f1']}
        history.append(row)
        print('epoch={epoch:03d} train_loss={train_loss:.4f} val_acc={val_accuracy:.3f} val_macro_f1={val_macro_f1:.3f}'.format(**row))
        if val_report['macro_f1'] > best + 1e-8:
            best = val_report['macro_f1']
            stale = 0
            torch.save({'model_state': model.state_dict(), 'labels': labels,
                        'frames': args.frames, 'version': 2,
                        'feature_spec': '33 MediaPipe-order x,y,z,visibility; hip-centered'}, out)
        else:
            stale += 1
        if stale >= args.patience:
            print(f'Early stopping at epoch {epoch}.')
            break
    with (report_dir / 'history.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader(); w.writerows(history)
    best_model, _ = load_model(out, device)
    report = {'validation': evaluate(best_model, val_dl, labels, device),
              'test': evaluate(best_model, test_dl, labels, device),
              'split': {'train': len(train_ids), 'validation': len(val_ids), 'test': len(test_ids),
                        'subject_disjoint': subjects is not None,
                        'train_subjects': int(len(np.unique(subjects[train_ids]))) if subjects is not None else None,
                        'validation_subjects': int(len(np.unique(subjects[val_ids]))) if subjects is not None else None,
                        'test_subjects': int(len(np.unique(subjects[test_ids]))) if subjects is not None else None},
              'note': 'Demo/synthetic measurements are NOT estimates of real-world performance.'}
    (report_dir / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Best model: {out}; evaluation: {report_dir / "evaluation.json"}; held-out test macro F1={report["test"]["macro_f1"]:.3f}')
    print('Measure real-world accuracy on unseen real people before deploying.')


def load_model(path, device='cpu'):
    # Only load checkpoints you created/trust; weights_only avoids arbitrary pickle objects.
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = ActionNet(len(ckpt['labels'])).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt


def predict(args):
    if not 0 <= args.threshold <= 1:
        raise ValueError('threshold must be between 0 and 1')
    model, meta = load_model(args.model)
    raw = np.load(args.input, allow_pickle=False)
    seq = raw['landmarks'] if isinstance(raw, np.lib.npyio.NpzFile) else raw
    if isinstance(raw, np.lib.npyio.NpzFile):
        raw.close()
    if seq.ndim == 4 and len(seq) == 1:
        seq = seq[0]
    seq = normalize(resample(seq, meta['frames']))
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.from_numpy(seq[None])), dim=1)[0].numpy()
    i = int(probabilities.argmax())
    result = {'action': meta['labels'][i] if float(probabilities[i]) >= args.threshold else 'unknown', 'confidence': round(float(probabilities[i]), 4),
              'probabilities': {k: round(float(v), 4) for k, v in zip(meta['labels'], probabilities)}}
    print(json.dumps(result, ensure_ascii=False, indent=2))


def demo(args):
    """Synthetic landmarks for pipeline smoke-testing only; not physiologically realistic."""
    rng = np.random.default_rng(args.seed)
    base = rng.uniform(0.35, 0.65, size=(33, 4)).astype('float32')
    base[:, 2] = 0
    base[:, 3] = 1
    base[[23, 24], :2] = [[0.47, 0.52], [0.53, 0.52]]
    base[[11, 12], :2] = [[0.45, 0.32], [0.55, 0.32]]
    X, y, subjects = [], [], []
    for subject in range(args.subjects):
        for cls in range(len(LABELS)):
            for take in range(args.takes):
                seq = np.repeat(base[None], args.frames, axis=0)
                phase = np.linspace(0, 2 * np.pi, args.frames) + rng.uniform(-0.25, 0.25)
                pulse = (1 - np.cos(phase)) * 0.5
                if cls == 0:
                    seq[:, [23, 24, 25, 26], 1] += pulse[:, None] * .20
                elif cls == 1:
                    seq[:, [11, 12, 13, 14], 1] += pulse[:, None] * .16
                elif cls == 2:
                    seq[:, [25, 27], 0] += pulse[:, None] * .20
                else:
                    seq[:, :, 1] += np.sin(phase)[:, None] * .003
                seq[:, :, :3] += rng.normal(0, 0.003, seq[:, :, :3].shape)
                seq[:, :, :2] += rng.normal(0, 0.01, (1, 1, 2))
                X.append(seq)
                y.append(cls)
                subjects.append(str(subject))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=np.stack(X), y=np.array(y), subjects=np.array(subjects))
    print(f'Created SYNTHETIC demo dataset: {args.out}; samples={len(y)}; subjects={args.subjects}')


def angle(a, b, c):
    a, b, c = np.asarray(a)[:2], np.asarray(b)[:2], np.asarray(c)[:2]
    v1, v2 = a - b, c - b
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return None if norm < 1e-8 else float(np.degrees(np.arccos(np.clip(v1 @ v2 / norm, -1, 1))))


class SquatCounter:
    """Example 2D knee-angle state machine; tune thresholds from real camera setup."""
    def __init__(self, down=100, up=160):
        if down >= up:
            raise ValueError('down must be less than up')
        self.down, self.up, self.state, self.reps = down, up, 'up', 0

    def reset_phase(self):
        self.state = 'up'

    def update(self, landmarks):
        p = np.asarray(landmarks)
        if p.shape != (33, 4):
            raise ValueError('One frame must be (33,4)')
        if min(p[[23, 25, 27, 24, 26, 28], 3]) < .5:
            return {'reps': self.reps, 'state': self.state, 'angle': None, 'status': 'low_visibility'}
        angles = [angle(p[h], p[k], p[a]) for h, k, a in [(23, 25, 27), (24, 26, 28)]]
        if None in angles:
            return {'reps': self.reps, 'state': self.state, 'angle': None, 'status': 'invalid_geometry'}
        knee = float(np.mean(angles))
        if self.state == 'up' and knee <= self.down:
            self.state = 'down'
        elif self.state == 'down' and knee >= self.up:
            self.state = 'up'
            self.reps += 1
        return {'reps': self.reps, 'state': self.state, 'angle': round(knee, 1), 'status': 'ok'}


def main():
    p = argparse.ArgumentParser(description='VisionFit local AI: keypoints -> action')
    sub = p.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('demo', help='Make fake data for code smoke test only')
    d.add_argument('--out', default='data/demo.npz')
    d.add_argument('--subjects', type=int, default=8)
    d.add_argument('--takes', type=int, default=8)
    d.add_argument('--frames', type=int, default=32)
    d.add_argument('--seed', type=int, default=42)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--labels', help='JSON file containing ordered class names')
    t.add_argument('--out', default='models/action.pt')
    t.add_argument('--frames', type=int, default=32)
    t.add_argument('--epochs', type=int, default=15)
    t.add_argument('--batch', type=int, default=32)
    t.add_argument('--lr', type=float, default=0.001)
    t.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--augment', action='store_true', help='Add small train-only landmark noise')
    t.add_argument('--class-weights', action='store_true', help='Weight rarer classes')
    t.add_argument('--patience', type=int, default=8, help='Early stopping epochs')
    t.add_argument('--val-fraction', type=float, default=0.2)
    t.add_argument('--test-fraction', type=float, default=0.2)
    t.add_argument('--report-dir', default='reports')
    q = sub.add_parser('predict')
    q.add_argument('--model', required=True)
    q.add_argument('--input', required=True, help='.npy (T,33,4) or .npz with landmarks')
    q.add_argument('--threshold', type=float, default=0.65, help='Below threshold -> unknown')
    args = p.parse_args()
    {'demo': demo, 'train': train, 'predict': predict}[args.cmd](args)


if __name__ == '__main__':
    main()
