"""Trainable, fully local 2D 33-landmark detector. No model downloads or external inference APIs.

Output is (33,4): image-normalized x,y; z=0 (not learned); visibility probability.
Requires YOUR labeled images to become useful. Not a pretrained detector.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

IMAGE_SIZE = 128
JOINTS = 33


class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.heatmap = nn.Conv2d(64, JOINTS, 1)
        self.visibility = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, JOINTS))

    def forward(self, x):
        z = self.features(x)
        return self.heatmap(z), self.visibility(z)


def decode_pose(heatmaps, visibility_logits):
    """Batch outputs into normalized xy and visibility. Argmax decoder for low compute."""
    if heatmaps.ndim != 4 or heatmaps.shape[1] != JOINTS:
        raise ValueError('Heatmaps must be (B,33,H,W)')
    b, _, h, w = heatmaps.shape
    loc = heatmaps.flatten(2).argmax(dim=-1)
    x = (loc % w).float() / max(w - 1, 1)
    y = (loc // w).float() / max(h - 1, 1)
    return torch.stack([x, y, torch.zeros_like(x), torch.sigmoid(visibility_logits)], dim=-1)


def pose_loss(heatmaps, visibility_logits, keypoints):
    """Supervised spatial Gaussian NLL + visibility BCE. Missing joints excluded from location loss."""
    b, joints, h, w = heatmaps.shape
    if keypoints.shape != (b, JOINTS, 3):
        raise ValueError('Keypoint annotations must be (B,33,3): normalized x,y,visible')
    target_vis = (keypoints[:, :, 2] >= 0.5).float()
    ys, xs = torch.meshgrid(torch.arange(h, device=heatmaps.device),
                            torch.arange(w, device=heatmaps.device), indexing='ij')
    coords_x = keypoints[:, :, 0, None, None] * (w - 1)
    coords_y = keypoints[:, :, 1, None, None] * (h - 1)
    dist2 = (xs[None, None] - coords_x).square() + (ys[None, None] - coords_y).square()
    target = torch.exp(-dist2 / (2 * 1.3**2)).flatten(2)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    location_nll = -(target * heatmaps.flatten(2).log_softmax(dim=-1)).sum(dim=-1)
    location_loss = (location_nll * target_vis).sum() / target_vis.sum().clamp_min(1)
    visibility_loss = nn.functional.binary_cross_entropy_with_logits(visibility_logits, target_vis)
    return location_loss + visibility_loss


def load_pose_dataset(path):
    with np.load(path, allow_pickle=False) as z:
        if not {'images', 'keypoints'}.issubset(z.files):
            raise ValueError('Expected images and keypoints arrays')
        images = z['images']
        pts = z['keypoints'].astype('float32')
        subjects = z['subjects'].astype(str) if 'subjects' in z.files else None
    if images.ndim != 4 or images.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3) or images.dtype != np.uint8:
        raise ValueError('images must be uint8 (N,128,128,3) RGB')
    if pts.shape != (len(images), JOINTS, 3) or len(images) < 6 or not np.isfinite(pts).all():
        raise ValueError('keypoints must be finite (N,33,3), N>=6')
    if ((pts[..., :2] < 0) | (pts[..., :2] > 1)).any() or ((pts[..., 2] < 0) | (pts[..., 2] > 1)).any():
        raise ValueError('keypoints x,y,visibility must be between 0 and 1')
    if subjects is not None and len(subjects) != len(images):
        raise ValueError('subjects length mismatch')
    return images, pts, subjects


def train(args):
    if args.epochs < 1 or args.batch < 1:
        raise ValueError('epochs/batch must be positive')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    images, points, subjects = load_pose_dataset(args.data)
    rng = np.random.default_rng(args.seed)
    if subjects is not None:
        units = np.unique(subjects)
        if len(units) < 3:
            raise ValueError('Need at least 3 different subjects for pose validation')
        perm = rng.permutation(units)
        count_val = max(1, round(len(units) * 0.2))
        val = np.flatnonzero(np.isin(subjects, perm[:count_val]))
        tr = np.flatnonzero(~np.isin(subjects, perm[:count_val]))
    else:
        print('WARNING: no subjects field. Random image split can leak identical people/videos.')
        perm = rng.permutation(len(images))
        count_val = max(1, round(len(images) * 0.2))
        val, tr = perm[:count_val], perm[count_val:]
    x = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float() / 255
    y = torch.from_numpy(points.copy())
    loader = DataLoader(TensorDataset(x[tr], y[tr]), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(x[val], y[val]), batch_size=args.batch)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    model = PoseNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float('inf')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = pose_loss(*model(bx), by)
            loss.backward()
            opt.step()
            train_sum += float(loss.item()) * len(by)
        model.eval()
        val_sum = 0.
        with torch.no_grad():
            for bx, by in val_loader:
                loss = pose_loss(*model(bx.to(device)), by.to(device))
                val_sum += float(loss.item()) * len(by)
        val_loss = val_sum / len(val)
        print(f'pose epoch={epoch:03d} train_loss={train_sum / len(tr):.3f} val_loss={val_loss:.3f}')
        if val_loss < best:
            best = val_loss
            torch.save({'state': model.state_dict(), 'image_size': IMAGE_SIZE, 'version': 1,
                        'z_prediction': False, 'note': 'Trained locally, 2D x/y only'}, args.out)
    print(f'Saved own pose model: {args.out}. Assess landmark accuracy on unseen real images.')


def load_pose_model(path, device='cpu'):
    ckpt = torch.load(path, weights_only=True, map_location=device)
    if ckpt.get('image_size') != IMAGE_SIZE:
        raise ValueError('Unsupported pose image size')
    model = PoseNet().to(device)
    model.load_state_dict(ckpt['state'])
    model.eval()
    return model


def infer_rgb(model, rgb_image, device='cpu'):
    """Input RGB uint8 (128,128,3); output (33,4), z=0 by design."""
    image = np.asarray(rgb_image)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError('RGB image must be uint8 (128,128,3)')
    inp = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32) / 255.
    with torch.no_grad():
        return decode_pose(*model(inp))[0].cpu().numpy().astype('float32')


def demo(args):
    """Render arbitrary colored-dot data ONLY for functionality testing, not real pose."""
    rng = np.random.default_rng(args.seed)
    images, pts, subjects = [], [], []
    for subject in range(args.subjects):
        for _ in range(args.takes):
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            points = np.zeros((JOINTS, 3), dtype='float32')
            # Toy dots have no correspondence to human anatomy.
            positions = rng.integers(8, IMAGE_SIZE - 8, size=(JOINTS, 2))
            for j, (x, y) in enumerate(positions):
                image[y - 2:y + 3, x - 2:x + 3, j % 3] = 255
                points[j] = [x / (IMAGE_SIZE - 1), y / (IMAGE_SIZE - 1), 1.]
            images.append(image); pts.append(points); subjects.append(f'demo_{subject}')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, images=np.stack(images), keypoints=np.stack(pts), subjects=np.asarray(subjects))
    print(f'Created SYNTHETIC dot pose dataset: {args.out}; not suitable for human inference.')


def main():
    parser = argparse.ArgumentParser(description='Train your OWN local 33-landmark pose model')
    sub = parser.add_subparsers(dest='command', required=True)
    d = sub.add_parser('demo', help='Synthetic dots; smoke-testing only')
    d.add_argument('--out', default='data/pose_demo.npz')
    d.add_argument('--subjects', type=int, default=5)
    d.add_argument('--takes', type=int, default=4)
    d.add_argument('--seed', type=int, default=42)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--out', default='models/pose.pt')
    t.add_argument('--epochs', type=int, default=30)
    t.add_argument('--batch', type=int, default=8)
    t.add_argument('--lr', type=float, default=0.001)
    t.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    t.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    {'demo': demo, 'train': train}[args.command](args)


if __name__ == '__main__':
    main()
