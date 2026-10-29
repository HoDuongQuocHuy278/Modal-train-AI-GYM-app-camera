"""Simple angle/hysteresis repetition counters. Not medical or expert coaching."""
import numpy as np
from visionfit import angle


class JointAngleCounter:
    def __init__(self, triplets, down=100, up=155, min_visibility=0.5, aggregate='mean'):
        if not 0 < down < up < 180:
            raise ValueError('Require 0 < down < up < 180')
        self.triplets = triplets
        self.down, self.up, self.visibility = down, up, min_visibility
        self.aggregate = aggregate
        self.state = 'up'
        self.reps = 0

    def reset_phase(self):
        """Preserve rep total; discard an unfinished repetition on activity switch."""
        self.state = 'up'

    def update(self, landmarks):
        pts = np.asarray(landmarks, dtype=np.float32)
        if pts.shape != (33, 4):
            raise ValueError('Expected one (33,4) frame')
        values = []
        for triplet in self.triplets:
            if min(pts[list(triplet), 3]) < self.visibility:
                continue
            value = angle(pts[triplet[0]], pts[triplet[1]], pts[triplet[2]])
            if value is not None:
                values.append(value)
        if not values:
            return self._result(None, 'low_visibility_or_invalid_geometry')
        value = float(min(values) if self.aggregate == 'min' else np.mean(values))
        if self.state == 'up' and value <= self.down:
            self.state = 'down'
        elif self.state == 'down' and value >= self.up:
            self.state = 'up'
            self.reps += 1
        return self._result(value, 'ok')

    def _result(self, value, status):
        return {'reps': self.reps, 'phase': self.state,
                'angle_degrees': round(value, 1) if value is not None else None,
                'status': status}


class PushUpCounter(JointAngleCounter):
    def __init__(self):
        super().__init__([(11, 13, 15), (12, 14, 16)], down=95, up=155)


class LungeCounter(JointAngleCounter):
    def __init__(self):
        # Use more-flexed visible knee. Tune for camera view before deployment.
        super().__init__([(23, 25, 27), (24, 26, 28)], down=100, up=155, aggregate='min')
