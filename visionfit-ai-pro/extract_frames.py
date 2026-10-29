"""Extract local video frames for offline manual pose annotation (no network)."""
import argparse
from pathlib import Path
from local_video import open_local_capture


def extract(video, output, every=10, limit=250):
    import cv2
    if every < 1 or limit < 1:
        raise ValueError('every and limit must be positive')
    cap = open_local_capture(video)
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    scanned = saved = 0
    try:
        while saved < limit:
            ok, frame = cap.read()
            if not ok:
                break
            scanned += 1
            if scanned % every:
                continue
            path = target / f'frame_{scanned:07d}.jpg'
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite {path}')
            if not cv2.imwrite(str(path), frame):
                raise OSError(f'Failed to write {path}')
            saved += 1
    finally:
        cap.release()
    print(f'Saved {saved} of {scanned} local video frames -> {target}')


def main():
    p = argparse.ArgumentParser(description='Local video -> images for annotation')
    p.add_argument('--video', required=True, help='Local path or camera index; no URLs')
    p.add_argument('--out', required=True)
    p.add_argument('--every', type=int, default=10)
    p.add_argument('--limit', type=int, default=250)
    a = p.parse_args()
    extract(a.video, a.out, a.every, a.limit)


if __name__ == '__main__':
    main()
