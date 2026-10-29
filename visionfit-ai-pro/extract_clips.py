"""Create action clips from offline video/webcam using YOUR local trained PoseNet."""
import argparse
from collections import deque
import json
from pathlib import Path
import re
import numpy as np
from pose_model import load_pose_model
from local_video import open_local_capture, landmarks_from_bgr, body_visible


def safe_name(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', value):
        raise ValueError('Class/subject identifiers must contain only A-Z a-z 0-9 _ -')
    return value


def extract(args):
    import cv2
    labels = json.loads(Path(args.labels).read_text(encoding='utf-8'))
    if args.label not in labels:
        raise ValueError(f'Unknown label {args.label!r}. Choose from {labels}')
    safe_name(args.label); safe_name(args.subject)
    if args.frames < 2 or args.hop < 1 or args.stride < 1 or args.max_frames < 0:
        raise ValueError('frames >=2, hop/stride >=1, max_frames >=0')
    model = load_pose_model(args.pose_model)
    cap = open_local_capture(args.source)
    stem = (re.sub(r'[^A-Za-z0-9_-]+', '_', Path(args.source).stem).strip('_')[:70] or 'clip') if not args.source.isdecimal() else 'webcam'
    target = Path(args.out_root) / args.label / args.subject
    target.mkdir(parents=True, exist_ok=True)
    window = deque(maxlen=args.frames)
    seen, valid, written, skips = 0, 0, 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and seen >= args.max_frames:
                break
            seen += 1
            if seen % args.stride:
                continue
            landmarks = landmarks_from_bgr(model, frame)
            if not body_visible(landmarks, args.min_body_visibility):
                window.clear()
                valid = 0
                skips += 1
                continue
            window.append(landmarks)
            valid += 1
            if len(window) == args.frames and (valid - args.frames) % args.hop == 0:
                dest = target / f'{stem}_{seen:08d}.npy'
                if dest.exists():
                    raise FileExistsError(f'Refusing to overwrite collected clip: {dest}')
                np.save(dest, np.stack(window).astype('float32'))
                written += 1
    finally:
        cap.release()
    print(f'Frames={seen}; windows={written}; invalid_pose_frames={skips}; folder={target}')
    if not written:
        print('No clips saved: check pose model quality, framing, thresholds and video length.')


def main():
    p = argparse.ArgumentParser(description='Local video -> own PoseNet -> labeled keypoint clips')
    p.add_argument('--source', required=True, help='Local mp4 path or camera integer (0)')
    p.add_argument('--pose-model', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--subject', required=True, help='Anonymous person ID, consistent across clips')
    p.add_argument('--labels', default='labels.json')
    p.add_argument('--out-root', default='data/action_clips')
    p.add_argument('--frames', type=int, default=32)
    p.add_argument('--hop', type=int, default=16)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--max-frames', type=int, default=1200, help='0 = unlimited')
    p.add_argument('--min-body-visibility', type=float, default=0.4)
    extract(p.parse_args())


if __name__ == '__main__':
    main()
