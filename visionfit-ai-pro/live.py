"""Offline webcam/local-video: own PoseNet -> own ActionNet -> smoothed class -> rep counter.
Not operational on real humans until TWO models have been trained on real labeled data.
"""
import argparse
from collections import deque
import json
import numpy as np
import torch
from local_video import open_local_capture, landmarks_from_bgr, body_visible, draw_skeleton
from pose_model import load_pose_model
from visionfit import SquatCounter, normalize, resample, load_model
from repetition import PushUpCounter, LungeCounter


class ActionSmoother:
    def __init__(self, labels, alpha=0.6, threshold=0.65, min_votes=2):
        if not 0 < alpha <= 1 or not 0 <= threshold <= 1 or min_votes < 1:
            raise ValueError('Invalid smoothing settings')
        self.labels = labels
        self.alpha, self.threshold, self.min_votes = alpha, threshold, min_votes
        self.probs, self.candidate, self.votes = None, None, 0
        self.current = 'unknown'

    def update(self, probabilities):
        probs = np.asarray(probabilities, dtype='float32')
        if probs.shape != (len(self.labels),):
            raise ValueError('Probability vector must match labels')
        self.probs = probs if self.probs is None else self.alpha * probs + (1 - self.alpha) * self.probs
        best = int(self.probs.argmax())
        new = self.labels[best] if self.probs[best] >= self.threshold else 'unknown'
        if new == self.candidate:
            self.votes += 1
        else:
            self.candidate, self.votes = new, 1
        if self.votes >= self.min_votes:
            self.current = new
        return self.current, float(self.probs[best])

    def clear(self):
        self.probs = None
        self.candidate = None
        self.votes = 0
        self.current = 'unknown'


def live(args):
    import cv2
    if args.step < 1 or args.max_frames < 0:
        raise ValueError('step >=1 and max_frames >=0')
    pose = load_pose_model(args.pose_model)
    action, meta = load_model(args.action_model)
    labels = meta['labels']
    frames = int(meta['frames'])
    cap = open_local_capture(args.source)
    window = deque(maxlen=frames)
    smoother = ActionSmoother(labels, args.alpha, args.threshold, args.min_votes)
    counters = {'squat': SquatCounter(), 'push_up': PushUpCounter(), 'lunge': LungeCounter()}
    stable_action = 'unknown'
    seen = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and seen >= args.max_frames:
                break
            seen += 1
            keypoints = landmarks_from_bgr(pose, frame)
            if not body_visible(keypoints, args.min_body_visibility):
                window.clear()
                smoother.clear()
                if stable_action in counters:
                    counters[stable_action].reset_phase()
                stable_action = 'unknown'
                status = 'NO RELIABLE POSE'
                confidence = 0.0
            else:
                window.append(keypoints)
                status = stable_action
                confidence = 0.0
                if len(window) == frames and seen % args.step == 0:
                    data = normalize(resample(np.stack(window), frames))
                    with torch.no_grad():
                        prob = torch.softmax(action(torch.from_numpy(data[None])), dim=1)[0].cpu().numpy()
                    label, confidence = smoother.update(prob)
                    if label != stable_action:
                        if stable_action in counters:
                            counters[stable_action].reset_phase()
                        stable_action = label
                    status = stable_action
                    info = {'frame': seen, 'action': stable_action, 'confidence': round(confidence, 4)}
                    print(json.dumps(info, ensure_ascii=False), flush=True)
                if stable_action in counters:
                    # Feed each frame for rep counting once activity is stable.
                    counters[stable_action].update(keypoints)
            if not args.headless:
                if body_visible(keypoints, args.min_body_visibility):
                    draw_skeleton(frame, keypoints, args.draw_visibility)
                reps = counters[stable_action].reps if stable_action in counters else 0
                cv2.putText(frame, f'VisionFit: {status} / conf {confidence:.2f}',
                            (12, 32), cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 240, 70), 2)
                cv2.putText(frame, f'Reps: {reps}   Q: quit', (12, 64),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 240, 70), 2)
                cv2.imshow('VisionFit local inference', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
    print(json.dumps({'processed_frames': seen, 'rep_totals': {name: c.reps for name, c in counters.items()}}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description='OFFLINE camera -> SELF-TRAINED pose + action models')
    p.add_argument('--source', default='0', help='Webcam index or existing local video file')
    p.add_argument('--pose-model', required=True, help='Self-trained PoseNet checkpoint')
    p.add_argument('--action-model', required=True)
    p.add_argument('--step', type=int, default=4)
    p.add_argument('--threshold', type=float, default=0.65)
    p.add_argument('--alpha', type=float, default=0.6)
    p.add_argument('--min-votes', type=int, default=2)
    p.add_argument('--min-body-visibility', type=float, default=0.4)
    p.add_argument('--draw-visibility', type=float, default=0.5)
    p.add_argument('--max-frames', type=int, default=0, help='0: until quit or video ends')
    p.add_argument('--headless', action='store_true', help='Print JSON, do not open a window')
    live(p.parse_args())


if __name__ == '__main__':
    main()
