"""ROS-independent policies for exact-stamp dynamic-pan BEV processing."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DynamicPanGuard:
    pan_min: float = -0.5236
    pan_max: float = 0.5236
    pan_tolerance: float = 0.001
    expected_tilt: float = -0.5236
    tilt_tolerance: float = 0.01

    def accepts(self, pan, tilt):
        pan, tilt = float(pan), float(tilt)
        return (math.isfinite(pan) and math.isfinite(tilt)
                and self.pan_min-self.pan_tolerance <= pan
                <= self.pan_max+self.pan_tolerance
                and abs(tilt-self.expected_tilt) <= self.tilt_tolerance)


class BoundedPendingFrames:
    """Keep the oldest request and the newest successor, never latest-TF."""

    def __init__(self, capacity=2):
        if capacity < 1:
            raise ValueError('capacity must be positive')
        self.capacity = int(capacity)
        self.items = []
        self.replaced = 0

    def append(self, frame, queued_at):
        entry = (frame, float(queued_at))
        if len(self.items) < self.capacity:
            self.items.append(entry)
        else:
            self.items[-1] = entry
            self.replaced += 1

    def expire(self, now, max_age):
        expired = 0
        while self.items and float(now)-self.items[0][1] > float(max_age):
            self.items.pop(0)
            expired += 1
        return expired

    def peek(self):
        return None if not self.items else self.items[0]

    def pop(self):
        return self.items.pop(0)

    def __len__(self):
        return len(self.items)
