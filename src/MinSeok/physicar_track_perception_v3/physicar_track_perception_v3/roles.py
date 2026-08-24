"""Role classification kept separate from path selection."""
from dataclasses import dataclass
import numpy as np
from .geometry import OrderedPolyline

CENTER, LEFT, RIGHT, UNKNOWN = 'CENTER', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'UNKNOWN'

@dataclass(frozen=True)
class Component:
    component_id: int
    color: str
    polyline: OrderedPolyline
    support: float

@dataclass(frozen=True)
class RoleConfig:
    center_colors: tuple = ('ORANGE',)
    center_max_abs_lateral: float = 0.12
    minimum_support: float = 0.20

def classify(component, config=RoleConfig()):
    if component.support < config.minimum_support:
        return UNKNOWN
    if component.color in config.center_colors and np.median(np.abs(component.polyline.points[:, 1])) <= config.center_max_abs_lateral:
        return CENTER
    # Boundary roles require an explicit geometric side; no persistent state.
    if float(np.median(component.polyline.points[:, 1])) > 0.0:
        return LEFT
    if float(np.median(component.polyline.points[:, 1])) < 0.0:
        return RIGHT
    return UNKNOWN
