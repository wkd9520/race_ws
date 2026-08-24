"""Small ordered metric-polyline primitives; no X sorting or polynomial fit."""
from dataclasses import dataclass
import numpy as np

def cumulative_s(points):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
        raise ValueError('points must have shape (N,2), N>=2')
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]

def canonical_order(points):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
        return np.empty((0, 2))
    # Ordering is supplied by the component graph.  Preserve it exactly;
    # neither endpoint heuristics nor X sorting are valid for 90-degree arcs.
    return p.copy()

def tangents(points):
    p = np.asarray(points, dtype=float)
    d = np.gradient(p, axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.maximum(n, 1e-12)

@dataclass(frozen=True)
class OrderedPolyline:
    points: np.ndarray
    s: np.ndarray
    support: float

    @classmethod
    def from_points(cls, points):
        ordered = canonical_order(points)
        s = cumulative_s(ordered)
        return cls(ordered, s, float(s[-1]))
