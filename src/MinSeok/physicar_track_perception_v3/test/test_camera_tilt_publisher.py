import math

import pytest

from physicar_track_perception_v3.camera_tilt_publisher import (
    DEFAULT_TILT_DEGREES,
    degrees_to_radians,
)


def test_default_tilt_commands_thirty_degrees_down():
    assert degrees_to_radians(DEFAULT_TILT_DEGREES) == pytest.approx(-math.pi / 6.0)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_tilt_is_rejected(value):
    with pytest.raises(ValueError):
        degrees_to_radians(value)
