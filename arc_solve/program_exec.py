import numpy as np
import numpy as np___
import traceback as traceback__
from scipy.ndimage import label as label__
from scipy.ndimage import label
import sys as sys___
import io as io___


def find_contiguous_shapes(grid, color):
    labeled_array, num_features = label__(grid == color)
    shapes = []
    for i in range(1, num_features + 1):
        shapes.append(np___.argwhere(labeled_array == i))
    return shapes


find_connected_components = find_contiguous_shapes


def find_bounding_box(shape):
    min_r = min(x for x, y in shape)
    max_r = max(x for x, y in shape)
    min_c = min(y for x, y in shape)
    max_c = max(y for x, y in shape)
    return (min_r, min_c, max_r, max_c)


def normalized_shape(shape):
    min_r, min_c, _, _ = find_bounding_box(shape)
    return np___.array(sorted((r - min_r, c - min_c) for r, c in shape))


def run_full_str(solution: str, inp: list[list[int]], catch_error: bool = True):
    sys___.stdout = io___.StringIO()
    sys___.stderr = io___.StringIO()
    sys___.stdout.close = lambda *args, **kwargs: None
    sys___.stderr.close = lambda *args, **kwargs: None

    globals_before__ = globals().copy()
    try:
        exec(solution, globals(), globals())
        out = transform(inp)
    except Exception as e:
        if not catch_error:
            raise e
        traceback__.print_exc(file=sys___.stderr)
        out = None  # TODO: we could return more info here if we wanted!!!
    finally:
        # slightly horrifying
        globals().update(globals_before__)
        for k in set(globals().keys()) - globals_before__.keys():
            del globals()[k]

        stdout = sys___.stdout.getvalue()
        stderr = sys___.stderr.getvalue()

        sys___.stdout = sys___.__stdout__
        sys___.stderr = sys___.__stderr__

    return out, stdout, stderr
