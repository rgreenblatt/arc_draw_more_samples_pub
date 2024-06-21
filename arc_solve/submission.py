from collections import defaultdict
from typing import TypeVar

import numpy as np
import attrs
from arc_solve.edit_distance import get_rank_geo_mean_score

from arc_solve.run_programs import RunOutput
from arc_solve.load_data import out_data_by_name_d


T = TypeVar("T")


def select_top_k_from_scored_dict(d: dict[T, float], k: int) -> list[T]:
    return [
        x
        for x, _ in sorted(
            d.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:k]
    ]


def select_best_to_submit(
    results: list[tuple[RunOutput, str]],
    train_data: list[dict[str, list[list[int]]]],
    test_inputs: list[list[list[int]]],
    n_to_submit: int = 3,
):
    filt_results = [
        (x, solution)
        for (x, solution) in results
        if x.all_test_output_valid()
        and all(
            test_output
            != test_input  # filter out cases where we try to return the input (this probably helps the output distance metric track what we want)
            for test_output, test_input in zip(x.test_results, test_inputs)
        )
    ]

    train_correct_valid_outputs = [
        x.test_output_as_hashable_unwrap()
        for x, _ in filt_results
        if x.all_train_correct()
    ]

    hashable_train_correct_output_to_score: dict[
        tuple[tuple[tuple[int, ...], ...], ...], float
    ] = defaultdict(float)

    for output in train_correct_valid_outputs:
        hashable_train_correct_output_to_score[output] += 1.0

    train_correct_selections = select_top_k_from_scored_dict(
        hashable_train_correct_output_to_score, n_to_submit
    )

    remaining_count = n_to_submit - len(train_correct_selections)

    other_items = [x for x, _ in filt_results if not x.all_train_correct()]

    if (
        len(other_items) == 0 or remaining_count == 0
    ):  # just for speedy early exit, probably doesn't speed up much in total
        return [[[list(z) for z in y] for y in x] for x in train_correct_selections]

    rank_prod_other_items = get_rank_geo_mean_score(
        [x.train_results for x in other_items],
        [x["output"] for x in train_data],
        make_further_lower=False,
    )

    assert len(rank_prod_other_items) == len(other_items)

    # select items where train outputs are close and we have many different items that result in that output
    # (e.g. majority vote style)
    hashable_output_to_sum_inv_rank_prod: dict[
        tuple[tuple[tuple[int, ...], ...], ...], float
    ] = defaultdict(float)

    for rank_prod, x in zip(rank_prod_other_items, other_items):
        hashable_output = x.test_output_as_hashable_unwrap()
        hashable_output_to_sum_inv_rank_prod[hashable_output] += 1.0 / rank_prod

    other_selections = select_top_k_from_scored_dict(
        hashable_output_to_sum_inv_rank_prod, remaining_count
    )

    out = train_correct_selections + other_selections

    out = [[[list(z) for z in y] for y in x] for x in out]

    return out


def make_submission_item_select_best_to_submit(
    name: str,
    results: list[tuple[RunOutput, str]],
    n_to_submit: int = 3,
):
    data = out_data_by_name_d[name]

    train_data = data["train"]
    test_inputs = [x["input"] for x in data["test"]]

    best_to_submit = select_best_to_submit(
        results, train_data=train_data, test_inputs=test_inputs, n_to_submit=n_to_submit
    )

    as_attempts = {f"attempt_{i+1}": x for i, x in enumerate(best_to_submit)}

    return [
        {k: v[which_inp] for k, v in as_attempts.items()}
        for which_inp in range(len(test_inputs))
    ]


def make_submission_dict(
    eval_out: dict[str, list[tuple[RunOutput, str]]],
    n_to_submit: int = 3,
):
    return {
        name: make_submission_item_select_best_to_submit(
            name, results, n_to_submit=n_to_submit
        )
        for name, results in eval_out.items()
    }


def score_submission_dict(
    submission: dict[str, list[dict[str, list[list[int]]]]],
    correct_outputs: dict[str, list[list[list[int]]]],
    allowed_attempts: int = 3,
):
    corr_by_name: dict[str, bool] = {}
    for name, expected in correct_outputs.items():
        sub = submission[name]
        assert len(sub) == len(expected)

        def get_warn(x, attempt, default):
            if attempt not in x:
                print("Missing!", name, attempt)
            return x.get(attempt, default)

        convert_subs_to_lists = [
            [get_warn(sub[i], f"attempt_{j+1}", None) for i in range(len(expected))]
            for j in range(allowed_attempts)
        ]

        corr = any(x == expected for x in convert_subs_to_lists)
        corr_by_name[name] = corr

    return corr_by_name, np.mean(list(corr_by_name.values()))
