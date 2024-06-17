from functools import partial
from typing import Optional, Sequence
import numpy as np


def simple_edit_distance(
    x: list[list[int]] | np.ndarray, y: list[list[int]] | np.ndarray
):
    if isinstance(x, np.ndarray):
        x_v = x
    else:
        x_v = np.array(x)
    if isinstance(y, np.ndarray):
        y_v = y
    else:
        y_v = np.array(y)

    if x_v.shape != y_v.shape:
        return 1.0

    return float(np.mean(x_v != y_v))


def is_valid(x: list[list[int]] | np.ndarray):
    try:
        if isinstance(x, np.ndarray):
            assert x.ndim == 2
            return True
        x_v = np.array(x)
        assert x_v.ndim == 2
        return True
    except (ValueError, AssertionError):
        return False


def maybe_invalid_edit_distance(
    source: Optional[list[list[int]] | np.ndarray],
    target: Optional[list[list[int]] | np.ndarray],
    is_further_good: bool = False,
):
    if target is None or not is_valid(target):
        return 1.0

    if source is None or not is_valid(source):
        return 0.0 if is_further_good else 1.0

    return simple_edit_distance(source, target)

def get_ranks(
    items: Sequence[Sequence[Optional[list[list[int]] | np.ndarray]]],
    to_item: Sequence[Optional[list[list[int]] | np.ndarray]],
    is_further_good: bool = False,
):
    edit_distances = np.array(
        [
            [
                maybe_invalid_edit_distance(x, b, is_further_good=is_further_good)
                for x, b in zip(xs, to_item)
            ]
            for xs in items
        ]
    )

    sort_idxs = np.argsort(edit_distances, axis=0)
    ranks = np.empty_like(sort_idxs)

    ranks[sort_idxs, np.arange(len(to_item))] = np.arange(len(items))[:, None] + 1

    return ranks


def make_valid_numpy_array(x: Optional[list[list[int]]]):
    if x is None:
        return None

    try:
        x_v = np.array(x)
        assert x_v.ndim == 2
        return x_v
    except (ValueError, AssertionError):
        return None


def geometric_mean(nums: np.ndarray, axis=None):
    # Check for any non-positive numbers
    assert (nums > 0).all()

    log_nums = np.log(nums)
    return np.exp(log_nums.mean(axis=axis))


def get_rank_geo_mean_score(
    items: Sequence[Sequence[Optional[list[list[int]] | np.ndarray]]],
    to_item: Sequence[Optional[list[list[int]] | np.ndarray]],
    make_further_lower: bool = False,
):
    raw_ranks = get_ranks(items, to_item, is_further_good=make_further_lower)

    if make_further_lower:
        ranks = len(items) - (raw_ranks - 1)
    else:
        ranks = raw_ranks

    # rank_prod = np.prod(ranks, axis=1)

    assert (ranks > 0).all()
    assert len(to_item) > 0
    rank_prod = geometric_mean(ranks, axis=1)

    assert rank_prod.shape == (len(items),)

    return rank_prod


def select_best_k_items_in_terms_of_distance(
    items: list[list[Optional[list[list[int]]]]],
    targets: list[list[list[int]]],
    k: int,
    farther_from: Optional[list[list[list[int]]]] = None,
    weight_on_further_from_best_so_far: float = 0.05,
    extra_scores: Optional[list[float]] = None,
    print_scores: bool = False,
):
    # TODO: try to equalize among a bunch of keys!!!

    # try to select items which are as close as possible to targets while being as far as possible from each other
    if len(items) == 0:
        return []

    items_as_numpy_arrays = [[make_valid_numpy_array(y) for y in x] for x in items]
    targets_as_numpy_arrays = [make_valid_numpy_array(y) for y in targets]

    rank_prod = get_rank_geo_mean_score(
        items_as_numpy_arrays, targets_as_numpy_arrays, make_further_lower=False
    )

    use_items: list[int] = []

    score = rank_prod.astype(float).copy()

    if farther_from is not None:
        ranks_farther_prod = get_rank_geo_mean_score(
            items_as_numpy_arrays, farther_from, make_further_lower=True
        )
        score += ranks_farther_prod.astype(float) * 0.25

    if extra_scores is not None:
        score += np.array(extra_scores)

    if print_scores:
        sorted_scores = np.copy(score)
        sorted_scores.sort()
        print(f"Initial scores: {sorted_scores[:20].tolist()=}")

    while len(use_items) < k:
        best = np.argmin(score)
        use_items.append(int(best))

        best_item = items_as_numpy_arrays[best]

        ranks_from_best_prod = get_rank_geo_mean_score(
            items_as_numpy_arrays, best_item, make_further_lower=True
        )

        score += (
            ranks_from_best_prod.astype(float) * weight_on_further_from_best_so_far / k
        )
        score[best] = float("inf")

    return use_items
