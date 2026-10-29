"""Build action dataset from labeled, locally produced landmark clips.
Folder convention: root/<action>/<anonymous_subject_id>/<clip>.npy or .npz.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from visionfit import resample


def build(root, label_file, out, frames=32):
    labels = json.loads(Path(label_file).read_text(encoding='utf-8'))
    if not isinstance(labels, list) or not labels or len(labels) != len(set(labels)):
        raise ValueError('labels.json must be a nonempty list of distinct labels')
    root = Path(root)
    clips, ids, subjects, records = [], [], [], []
    for i, name in enumerate(labels):
        folder = root / name
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing action folder {folder}')
        for file in sorted(folder.glob('*/*')):
            if file.suffix.lower() not in ('.npy', '.npz') or not file.is_file():
                continue
            raw = np.load(file, allow_pickle=False)
            if isinstance(raw, np.lib.npyio.NpzFile):
                try:
                    seq = raw['landmarks']
                finally:
                    raw.close()
            else:
                seq = raw
            if seq.ndim != 3 or seq.shape[1:] != (33, 4) or len(seq) < 2 or not np.isfinite(seq).all():
                raise ValueError(f'Invalid (T,33,4) clip: {file}')
            clips.append(resample(seq, frames))
            ids.append(i)
            subjects.append(file.parent.name)
            records.append(str(file))
    if not clips or set(ids) != set(range(len(labels))):
        raise ValueError('Every class must contain at least one valid clip')
    if len(set(subjects)) < 5:
        raise ValueError('At least five distinct anonymous subjects are required for subject-disjoint train/val/test')
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, X=np.stack(clips).astype('float32'),
                        y=np.asarray(ids, dtype='int64'), subjects=np.asarray(subjects))
    print(f'Saved {len(clips)} clips from {len(set(subjects))} subjects to {destination}')
    print(json.dumps({label: ids.count(i) for i, label in enumerate(labels)}, indent=2))
    return destination


def main():
    p = argparse.ArgumentParser(description='Aggregate local action keypoint files')
    p.add_argument('--root', default='data/action_clips')
    p.add_argument('--labels', default='labels.json')
    p.add_argument('--out', default='data/action_real.npz')
    p.add_argument('--frames', type=int, default=32)
    args = p.parse_args()
    build(args.root, args.labels, args.out, args.frames)


if __name__ == '__main__':
    main()
