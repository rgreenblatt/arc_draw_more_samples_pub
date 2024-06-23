import re
import json
import traceback
from typing import Any, Optional

import numpy as np
import attrs
from cattrs.preconf.json import make_converter
from arc_solve.edit_distance import is_valid

from arc_solve.multiprocessing_with_timeout import map_fixed_func_with_timeout
from arc_solve.program_exec import run_full_str
from arc_solve.load_data import out_data_by_name_d
from rrutils.redis_cache_wrapper import RedisWrapper, cache_key


@attrs.frozen
class KeyNameS:
    key: str
    name: str
    s: str
    s_idx: int


@attrs.frozen
class StdoutStderr:
    stdout: str
    stderr: str


@attrs.frozen
class RunOutputHashable:
    train_results: tuple[tuple[tuple[int, ...], ...] | None, ...]
    train_stdout_stderr: tuple[StdoutStderr, ...]
    train_corr: tuple[bool, ...]
    test_results: tuple[tuple[tuple[int, ...], ...] | None, ...]
    test_stdout_stderr: tuple[StdoutStderr, ...]


@attrs.frozen
class RunOutput:
    train_results: list[list[list[int]] | None]
    train_stdout_stderr: list[StdoutStderr]
    train_corr: list[bool]
    test_results: list[list[list[int]] | None]
    test_stdout_stderr: list[StdoutStderr]

    def fraction_train_correct(self):
        return sum(self.train_corr) / len(self.train_corr)

    def all_train_correct(self):
        return all(self.train_corr)

    def test_output_as_hashable(self):
        return tuple(
            [
                tuple(tuple(y) for y in x) if x is not None else None
                for x in self.test_results
            ]
        )

    def test_output_as_hashable_unwrap(self):
        out: list[tuple[tuple[int, ...], ...]] = []
        for x in self.test_output_as_hashable():
            assert x is not None
            out.append(x)

        return tuple(out)

    def all_test_output_valid(self):
        return all(x is not None and is_valid(x) for x in self.test_results)

    def to_hashable(self):
        return RunOutputHashable(
            train_results=tuple(
                tuple(tuple(y) for y in x) if x is not None else None
                for x in self.train_results
            ),
            train_stdout_stderr=tuple(self.train_stdout_stderr),
            train_corr=tuple(self.train_corr),
            test_results=tuple(
                tuple(tuple(y) for y in x) if x is not None else None
                for x in self.test_results
            ),
            test_stdout_stderr=tuple(self.test_stdout_stderr),
        )

    @classmethod
    def from_hashable(cls, x: RunOutputHashable):
        return cls(
            train_results=[
                [list(y) for y in x] if x is not None else None for x in x.train_results
            ],
            train_stdout_stderr=list(x.train_stdout_stderr),
            train_corr=list(x.train_corr),
            test_results=[
                [list(y) for y in x] if x is not None else None for x in x.test_results
            ],
            test_stdout_stderr=list(x.test_stdout_stderr),
        )


@attrs.frozen
class RunItem:
    key_name_s: KeyNameS
    run_output: RunOutput

    def __attrs_post_init__(self):
        assert self.run_output is not None

    @classmethod
    def from_maybe_none(
        cls,
        key_name_s: KeyNameS,
        run_output: Optional[RunOutput],
        n_train: int,
        n_test: int,
    ):
        if run_output is None:
            run_output_final = RunOutput(
                train_results=[None] * n_train,
                train_stdout_stderr=[StdoutStderr(stdout="", stderr="TIMEOUT")]
                * n_train,
                train_corr=[False] * n_train,
                test_results=[None] * n_test,
                test_stdout_stderr=[StdoutStderr(stdout="", stderr="TIMEOUT")] * n_test,
            )
        else:
            run_output_final = run_output

        return cls(key_name_s=key_name_s, run_output=run_output_final)


async def evaluate_funcs_with_timeout_cache(
    items: list[KeyNameS], timeout: float = 2.0, skip_cache: bool = False
):
    out = await map_fixed_func_with_timeout(
        run_on_train_test,
        [
            (
                (
                    x.name,
                    x.s,
                ),
                {},
            )
            for x in items
        ],
        timeout=timeout,
        wrap_call=(
            None if skip_cache else (evaluate_cache_try_get, evaluate_cache_try_set)
        ),
    )

    return [
        RunItem.from_maybe_none(
            key_name_s=x,
            run_output=y,
            n_train=len(out_data_by_name_d[x.name]["train"]),
            n_test=len(out_data_by_name_d[x.name]["test"]),
        )
        for x, y in zip(items, out)
    ]


key_name = "run_arc_programs_24"

json_converter = make_converter()


async def evaluate_cache_try_get(*args, **kwargs):
    key = cache_key(json.dumps([args, kwargs]), key_name)
    cached_res = await RedisWrapper.singleton().read(key)
    if cached_res is not None:
        return True, json_converter.loads(cached_res["value"], Optional[RunOutput])

    return False, None


async def evaluate_cache_try_set(out, *args, **kwargs):
    key = cache_key(json.dumps([args, kwargs]), key_name)
    await RedisWrapper.singleton().write(key, {"value": json_converter.dumps(out)})


noop_code = """
def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    raise NotImplementedError()
""".strip()


def clean_code(s: str):
    return s.replace("\t", " " * 4)


def parse_python_backticks(s: str):
    if s.count("```python") == 0:
        print(f"NO CODE BLOCKS")
        out = s.partition("</reasoning>")[2]
        if out == "":
            return noop_code
        return clean_code(out)

    if s.count("```python") > 1:
        # print(f"MULTIPLE CODE BLOCKS\n=====\n\n{s}\n\n=====")
        s = "```python" + s.split("```python")[-1]

    assert s.count("```python") == 1

    attempted_search = re.search(r"```python\n(.*)\n```", s, re.DOTALL | re.MULTILINE)
    if attempted_search is not None:
        return clean_code(attempted_search.group(1))

    attempted_search = re.search(r"```python\n(.*)\n`", s, re.DOTALL | re.MULTILINE)
    if attempted_search is not None:
        print(f"PARSE ERROR CASE (1)")
        return clean_code(attempted_search.group(1))
    else:
        print(f"PARSE ERROR CASE (2!)")

    return clean_code(s.partition("```python")[2])


def munge_process_output(x: tuple[Any, str, str]):
    val, stdout, stderr = x

    out_str = StdoutStderr(stdout, stderr)

    if val is None:
        return None, out_str

    try:
        out_array = [[int(y) for y in z] for z in val]

        arr = np.array(out_array)
        if arr.ndim != 2 or arr.shape[0] > 100 or arr.shape[1] > 100:
            return None, out_str

        return out_array, out_str
    except (ValueError, TypeError):
        return None, out_str


def run_on_train_test(name: str, s: str):
    data = out_data_by_name_d[name]
    train = data["train"]
    test = data["test"]

    try:
        solution = parse_python_backticks(s)
    except Exception as e:
        print("FAIL!", e)
        traceback.print_exc()
        raise

    # print(s)
    # print(repr(s))

    train_results_both = [
        munge_process_output(run_full_str(solution, x["input"], catch_error=True))
        for x in train
    ]
    train_results = [x for x, _ in train_results_both]
    train_stdout_stderr = [y for _, y in train_results_both]
    train_corr = [x == y for x, y in zip(train_results, [x["output"] for x in train])]

    test_results_both = [
        munge_process_output(run_full_str(solution, x["input"], catch_error=True))
        for x in test
    ]
    test_results = [x for x, _ in test_results_both]
    test_stdout_stderr = [y for _, y in test_results_both]

    return RunOutput(
        train_results=train_results,
        train_stdout_stderr=train_stdout_stderr,
        train_corr=train_corr,
        test_results=test_results,
        test_stdout_stderr=test_stdout_stderr,
    )
