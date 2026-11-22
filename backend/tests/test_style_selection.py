import numpy as np
import pytest

from photo_advisor.style_selection import diverse_indices


def tile(lightness):
    return np.full((3, 3, 3), [lightness, 0, 0], dtype=np.float32)


@pytest.mark.parametrize("values, limit, expected", [
    ([], 4, []), ([50], 4, [0]), ([50, 50, 50], 4, [0]),
    ([25, 50, 75], 2, [0, 2]), ([25, 25, 75], 4, [0, 2]),
])
def test_style_selection(values, limit, expected):
    assert diverse_indices([tile(value) for value in values], limit) == expected
