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

    if len(other_items) == 0 or remaining_count == 0: # just for speedy early exit, probably doesn't speed up much in total
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


def is_correct_select_best_to_submit(
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

    # actually grade here!
    test_data = data["test"]
    test_data_outputs = [x["output"] for x in test_data]
    return any(x == test_data_outputs for x in best_to_submit)


def is_correct_select_best_on_dict(
    eval_out: dict[str, list[tuple[RunOutput, str]]],
    n_to_submit: int = 3,
):
    return {
        name: is_correct_select_best_to_submit(name, results, n_to_submit=n_to_submit)
        for name, results in eval_out.items()
    }


@attrs.frozen
class SubInfo:
    is_corr: bool
    run_output: RunOutput
    reasoning_and_code: str
    weight: float

    @classmethod
    def from_tup(cls, is_corr: bool, rest: tuple[RunOutput, str, float]):
        run_output, reasoning_and_code, weight = rest

        return cls(
            is_corr=is_corr,
            run_output=run_output,
            reasoning_and_code=reasoning_and_code,
            weight=weight,
        )


def is_correct_select_best_to_submit_and_get_results_for_each(
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

    hashable_test_outputs_to_solution_weight: dict[
        tuple[tuple[tuple[int, ...], ...], ...], list[tuple[RunOutput, str, float]]
    ] = defaultdict(list)

    for x, solution in results:
        try:
            test_output_hashable = x.test_output_as_hashable_unwrap()
        except AssertionError:
            continue
        hashable_test_outputs_to_solution_weight[test_output_hashable].append(
            (
                x,
                solution,
                1e20 if x.all_train_correct() else 1.0,
            )  # 1e20 is a hack to get a submission which is correct on train set (if exists)
        )

    gen = np.random.default_rng(42)

    normalize_list = lambda x: np.array(x) / np.sum(x)

    hashable_test_outputs_to_sampled_solution = {
        k: tuple(
            gen.choice(
                np.array(v), size=1, p=normalize_list([w for _, _, w in v]), axis=0
            )[0].tolist()
        )
        for k, v in hashable_test_outputs_to_solution_weight.items()
    }

    # actually grade here!
    test_data = data["test"]
    test_data_outputs = [x["output"] for x in test_data]
    return [
        SubInfo.from_tup(
            x == test_data_outputs,
            hashable_test_outputs_to_sampled_solution[
                tuple(tuple(tuple(a) for a in b) for b in x)
            ],
        )
        for x in best_to_submit
    ]


def submission_nice_info_by_problem(
    eval_out: dict[str, list[tuple[RunOutput, str]]],
    n_to_submit: int = 3,
):
    return {
        k: is_correct_select_best_to_submit_and_get_results_for_each(
            k, v, n_to_submit=n_to_submit
        )
        for k, v in eval_out.items()
    }


def mean_correct_select_best_on_dict(
    eval_out: dict[str, list[tuple[RunOutput, str]]],
    n_to_submit: int = 3,
):
    correct_dict = is_correct_select_best_on_dict(eval_out, n_to_submit=n_to_submit)
    return sum(correct_dict.values()) / len(correct_dict)


def each_mean_correct_select_best_on_dict(
    all_eval_out: dict[str, dict[str, list[tuple[RunOutput, str]]]],
    n_to_submit: int = 3,
):
    return {
        k: mean_correct_select_best_on_dict(v, n_to_submit=n_to_submit)
        for k, v in all_eval_out.items()
    }
