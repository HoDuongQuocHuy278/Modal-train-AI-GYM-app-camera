"""Offline OpenCV manual labeling for PoseNet: 33 ordered clicks per image.
Click a visible landmark; X marks it invisible, U undo, S save when done, Q quit.
Images/coordinates never leave the machine. Requires opencv-python.
"""
import argparse
from pathlib import Path
import numpy as np
from pose_model import IMAGE_SIZE, JOINTS

NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner',
    'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
    'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
    'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
    'right_heel', 'left_foot_index', 'right_foot_index',
]


def annotate(image_dir, destination, subject, append=False):
    import cv2  # optional local package, not a remote API
    if not subject or '/' in subject or '\\' in subject:
        raise ValueError('Use a short anonymous subject ID without slashes')
    files = sorted(f for f in Path(image_dir).iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
    if not files:
        raise FileNotFoundError('No local images found for annotation')
    path = Path(destination)
    if path.exists() and not append:
        raise FileExistsError('Dataset exists; pass --append to add samples without losing data')
    images, keypoints, subjects = [], [], []
    if path.exists():
        with np.load(path, allow_pickle=False) as d:
            images, keypoints, subjects = list(d['images']), list(d['keypoints']), list(d['subjects'])
    cv2.namedWindow('VisionFit Pose Labeler', cv2.WINDOW_NORMAL)
    for file in files:
        original = cv2.imread(str(file))
        if original is None:
            print(f'Skipping unreadable image: {file}')
            continue
        frame = cv2.resize(original, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        clicks = []

        def on_mouse(event, x, y, flags, userdata):
            if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < JOINTS:
                # Display is nearest-neighbor enlarged 4x; screen coords 0..511.
                clicks.append((min(max(x / (IMAGE_SIZE * 4 - 1), 0.), 1.),
                               min(max(y / (IMAGE_SIZE * 4 - 1), 0.), 1.), 1.))

        cv2.setMouseCallback('VisionFit Pose Labeler', on_mouse)
        while True:
            display = cv2.resize(frame, (IMAGE_SIZE * 4, IMAGE_SIZE * 4), interpolation=cv2.INTER_NEAREST)
            for idx, (x, y, visible) in enumerate(clicks):
                if visible:
                    cv2.circle(display, (round(x * 511), round(y * 511)), 4, (50, 240, 50), -1)
                    cv2.putText(display, str(idx), (round(x * 511) + 4, round(y * 511)),
                                cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
            name = NAMES[len(clicks)] if len(clicks) < JOINTS else 'DONE - press S to save'
            cv2.rectangle(display, (0, 0), (512, 53), (0, 0, 0), -1)
            cv2.putText(display, f'{file.name} [{len(clicks)}/33] {name}', (4, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            cv2.putText(display, 'CLICK=point X=hidden U=undo S=save Q=quit', (4, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1)
            cv2.imshow('VisionFit Pose Labeler', display)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), 27):
                cv2.destroyAllWindows()
                _save(path, images, keypoints, subjects)
                print(f'Stopped. Total saved: {len(images)} -> {path}')
                return
            if key == ord('x') and len(clicks) < JOINTS:
                clicks.append((0., 0., 0.))
            if key == ord('u') and clicks:
                clicks.pop()
            if key == ord('s') and len(clicks) == JOINTS:
                images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                keypoints.append(np.asarray(clicks, dtype='float32'))
                subjects.append(subject)
                print(f'Saved: {file.name}')
                break
    cv2.destroyAllWindows()
    _save(path, images, keypoints, subjects)
    print(f'Total saved: {len(images)} -> {path}')


def _save(path, images, keypoints, subjects):
    if not images:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=np.asarray(images, dtype=np.uint8),
                        keypoints=np.asarray(keypoints, dtype='float32'), subjects=np.asarray(subjects))


def main():
    parser = argparse.ArgumentParser(description='Manually annotate 33 points in LOCAL images')
    parser.add_argument('--images', required=True)
    parser.add_argument('--out', default='data/pose_real.npz')
    parser.add_argument('--subject', required=True)
    parser.add_argument('--append', action='store_true')
    args = parser.parse_args()
    annotate(args.images, args.out, args.subject, args.append)


if __name__ == '__main__':
    main()
