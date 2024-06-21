# %%
# import os

# # %%

# os.environ["REDIS_READER_PORT"] = "6381"
# os.environ["INPUT_JSON"] = "./arc-agi_evaluation_challenges.json"
# os.environ["INPUT_TRAIN_JSON"] = "./arc-agi_training_challenges.json"

# os.environ["BAN_UNCACHED_LLM_QUERIES"] = "1"

# %%

# %%


from collections import defaultdict
import os
import json
import random
from typing import Any, Callable, Optional, TypeVar
import asyncio


from tqdm.asyncio import tqdm_asyncio
import nest_asyncio
import numpy as np
from cattrs.preconf.json import make_converter

json_converter = make_converter()

from arc_solve.prompting import (
    DisplayArgs,
    PromptArgs,
    fix_on_input,
    is_eq_size_item,
    run_on_input_with_name_alt,
)
from arc_solve.submission import (
    make_submission_dict,
    score_submission_dict,
)
from arc_solve.render import RenderArgs, grid_to_base64_png_oai_content, show_grid
from arc_solve.run_programs import (
    KeyNameS,
    RunItem,
    RunOutput,
    StdoutStderr,
    evaluate_funcs_with_timeout_cache,
)
from arc_solve.edit_distance import is_valid, select_best_k_items_in_terms_of_distance
from arc_solve.load_data import (
    out_data_by_name_d,
    loaded_names,
)
from arc_solve.reasoning_and_labels import (
    code_repair_example_4,
    code_repair_reasoning_examples_change_alt_color,
    code_repair_reasoning_examples_change_alt_color_new_long,
    code_repair_reasoning_examples_change_alt_color_new_long_use_diff,
    code_repair_reasoning_examples_change_alt_color_new_short,
    reasoning_labeled_items_alt,
    reasoning_labeled_items,
    reasoning_labeled_items_ascii,
    reasoning_labeled_items_full_spreadsheet_alt_color_fresh_hard,
    reasoning_labeled_items_full_spreadsheet_alt_color_fresh_hard_alt,
    code_repair_reasoning_examples,
    code_repair_reasoning_examples_use_diff,
    code_repair_reasoning_examples_multi,
    reasoning_labeled_items_full_spreadsheet_alt_color,
    code_repair_example_12_for_spreadsheet_alt_color,
    reasoning_labeled_items_full_spreadsheet_alt_color_extra,
    reasoning_labeled_items_full_spreadsheet_alt_color_extra_extra,
    reasoning_labeled_items_full_spreadsheet_alt_color_alt,
    reasoning_labeled_items_full_spreadsheet_alt_color_alt_again,
    reasoning_labeled_change_spreadsheet_prompt_alt_color_add,
    reasoning_labeled_change_spreadsheet_prompt_alt_color_add_swap,
    reasoning_labeled_change_spreadsheet_prompt_alt_color_add_just_change,
    reasoning_labeled_items_alt_color,
    reasoning_labeled_change_prompt_alt_color,
    reasoning_labeled_change_prompt_alt_color_add,
    reasoning_labeled_change_prompt_alt_color_add_swap,
    reasoning_labeled_change_prompt_alt_color_add_swap_again,
    reasoning_labeled_change_prompt_alt_color_add_just_change,
    code_repair_spreadsheet_alt_color_reasoning_examples,
    code_repair_spreadsheet_alt_color_reasoning_examples_swap,
    code_repair_example_3,
    code_repair_spreadsheet_alt_color_reasoning_examples_alt_shorter,
    reasoning_labeled_change_spreadsheet_prompt_alt_color,
    reasoning_labeled_change_prompt_alt_color_add_swap_minor_alt,
    reasoning_labeled_change_prompt_alt_color_total_alternative_prompt,
    reasoning_labeled_change_prompt_alt_color_another_alt_prompt,
    reasoning_labeled_items_full_spreadsheet_alt_color_concise_diff,
    # code_repair_spreadsheet_w_diff_new_alt_color_reasoning_examples,
)

nest_asyncio.apply()


# %%

eval_out_here = asyncio.run(
    evaluate_funcs_with_timeout_cache(
        [
            KeyNameS(
                key="labels",
                name=name,
                s=s,
                s_idx=-10,
            )
            for name, s in reasoning_labeled_items
            + reasoning_labeled_items_alt
            + reasoning_labeled_items_full_spreadsheet_alt_color
            + reasoning_labeled_items_full_spreadsheet_alt_color_alt
            + reasoning_labeled_items_alt_color
            + reasoning_labeled_change_prompt_alt_color
            + reasoning_labeled_change_prompt_alt_color_add
            + reasoning_labeled_change_prompt_alt_color_add_swap
            + reasoning_labeled_change_prompt_alt_color_add_just_change
            + reasoning_labeled_change_spreadsheet_prompt_alt_color
            + reasoning_labeled_change_spreadsheet_prompt_alt_color_add
            + reasoning_labeled_change_spreadsheet_prompt_alt_color_add_swap
            + reasoning_labeled_change_spreadsheet_prompt_alt_color_add_just_change
            + reasoning_labeled_items_full_spreadsheet_alt_color_fresh_hard
            + reasoning_labeled_items_full_spreadsheet_alt_color_fresh_hard_alt
            + reasoning_labeled_change_prompt_alt_color_add_swap_minor_alt
            + reasoning_labeled_change_prompt_alt_color_total_alternative_prompt
            + reasoning_labeled_items_full_spreadsheet_alt_color_concise_diff
            + reasoning_labeled_change_prompt_alt_color_another_alt_prompt
        ],
        timeout=5.0,
    )
)
# len(eval_out_here)
for item in eval_out_here:
    assert item.run_output is not None
    assert item.run_output.all_train_correct(), f"fail at {item.key_name_s.name}"

# %%

code_repair_reasoning_examples_use = (
    # code_repair_reasoning_examples
    code_repair_reasoning_examples_use_diff
)

# %%

code_reasoning_examples_all_outputs = asyncio.run(
    evaluate_funcs_with_timeout_cache(
        [
            KeyNameS(
                key=f"code_reasoning_step_idx_{idx}",
                name=name,
                s=s,
                s_idx=idx,
            )
            for name, fixs in code_repair_reasoning_examples_use
            for idx, s in enumerate(fixs)
        ],
        timeout=5.0,
    )
)

code_reasoning_examples_name_to_idx: dict[str, dict[int, RunOutput]] = defaultdict(dict)

for item in code_reasoning_examples_all_outputs:
    code_reasoning_examples_name_to_idx[item.key_name_s.name][
        item.key_name_s.s_idx
    ] = item.run_output

for k, vs in code_reasoning_examples_name_to_idx.items():
    last = vs[max(vs.keys())]
    assert last.all_train_correct()

# %%

code_repair_reasoning_examples_w_outputs = [
    (
        name,
        [
            (s, code_reasoning_examples_name_to_idx[name][idx])
            for idx, s in enumerate(reasoning)
        ],
    )
    for name, reasoning in code_repair_reasoning_examples_use
]

# %%

code_repair_spreadsheet_alt_color_reasoning_examples_use = (
    # code_repair_spreadsheet_alt_color_reasoning_examples
    code_repair_spreadsheet_alt_color_reasoning_examples_swap  # IDK if we should actually use this...
    # code_repair_spreadsheet_alt_color_reasoning_examples_alt_shorter
)

code_reasoning_examples_spreadsheet_all_outputs = asyncio.run(
    evaluate_funcs_with_timeout_cache(
        [
            KeyNameS(
                key=f"code_reasoning_spreadsheet_step_idx_{idx}",
                name=name,
                s=s,
                s_idx=idx,
            )
            for name, fixs in code_repair_spreadsheet_alt_color_reasoning_examples_use
            for idx, s in enumerate(fixs)
        ],
        timeout=5.0,
    )
)

code_reasoning_examples_spreadsheet_name_to_idx: dict[str, dict[int, RunOutput]] = (
    defaultdict(dict)
)

for item in code_reasoning_examples_spreadsheet_all_outputs:
    code_reasoning_examples_spreadsheet_name_to_idx[item.key_name_s.name][
        item.key_name_s.s_idx
    ] = item.run_output

for k, vs in code_reasoning_examples_spreadsheet_name_to_idx.items():
    last = vs[max(vs.keys())]
    assert last.all_train_correct()


# %%

code_repair_reasoning_examples_w_outputs_spreadsheet_alt_color = [
    (
        name,
        [
            (s, code_reasoning_examples_spreadsheet_name_to_idx[name][idx])
            for idx, s in enumerate(reasoning)
        ],
    )
    for name, reasoning in code_repair_spreadsheet_alt_color_reasoning_examples_use
]

# %%

code_repair_reasoning_examples_change_alt_color_use = (
    # code_repair_reasoning_examples_change_alt_color
    # code_repair_reasoning_examples_change_alt_color_new_long
    # code_repair_reasoning_examples_change_alt_color_new_short
    code_repair_reasoning_examples_change_alt_color_new_long_use_diff
)

# %%

code_reasoning_examples_change_alt_color_all_outputs = asyncio.run(
    evaluate_funcs_with_timeout_cache(
        [
            KeyNameS(
                key=f"code_reasoning_change_alt_color_step_idx_{idx}",
                name=name,
                s=s,
                s_idx=idx,
            )
            for name, fixs in code_repair_reasoning_examples_change_alt_color_use
            for idx, s in enumerate(fixs)
        ],
        timeout=5.0,
    )
)

code_reasoning_examples_change_alt_color_name_to_idx: dict[
    str, dict[int, RunOutput]
] = defaultdict(dict)

for item in code_reasoning_examples_change_alt_color_all_outputs:
    code_reasoning_examples_change_alt_color_name_to_idx[item.key_name_s.name][
        item.key_name_s.s_idx
    ] = item.run_output

for k, vs in code_reasoning_examples_change_alt_color_name_to_idx.items():
    last = vs[max(vs.keys())]
    assert last.all_train_correct()

# %%

code_repair_reasoning_examples_w_outputs_change_alt_color = [
    (
        name,
        [
            (s, code_reasoning_examples_change_alt_color_name_to_idx[name][idx])
            for idx, s in enumerate(reasoning)
        ],
    )
    for name, reasoning in code_repair_reasoning_examples_change_alt_color_use
]

# %%

names_alt = list(loaded_names)

if "RUN_ON_SUBSET" in os.environ:
    subset = int(os.environ["RUN_ON_SUBSET"])
    names_alt = names_alt[:subset]

len(names_alt)

# %%

prompt_settings = [
    # V0
    # PromptArgs(
    #     name="default", display_args=DisplayArgs(), legacy_color_to_index=True
    # ),
    # # V1.0
    # PromptArgs(
    #     name="use_spreadsheet_or_change_prompt",
    #     display_args=DisplayArgs(
    #         spreadsheet_ascii=True,
    #         spreadsheet_ascii_full=True,
    #         render_args=RenderArgs(
    #             use_alt_color_scheme=True,
    #         ),
    #     ),
    #     use_spreadsheet_if_eq_size_and_change_prompt_otherwise=True,
    # ),
    # # V1.1
    # PromptArgs(
    #     name="use_spreadsheet_or_change_prompt_alts",
    #     display_args=DisplayArgs(
    #         spreadsheet_ascii=True,
    #         spreadsheet_ascii_full=True,
    #         render_args=RenderArgs(
    #             use_alt_color_scheme=True,
    #         ),
    #         max_allowed_tokens_full_ascii_grid=300,
    #     ),
    #     force_reasoning_labeled_items_spreadsheet_ascii=tuple(
    #         reasoning_labeled_items_full_spreadsheet_alt_color_fresh_hard_alt
    #     ),
    #     force_reasoning_labeled_items_change_prompt=tuple(
    #         reasoning_labeled_change_prompt_alt_color_total_alternative_prompt
    #     ),
    #     use_spreadsheet_if_eq_size_and_change_prompt_otherwise=True,
    # ),
    # V2
    PromptArgs(
        name="use_spreadsheet_or_change_concise_diff",
        display_args=DisplayArgs(
            spreadsheet_ascii=True,
            spreadsheet_ascii_full=True,
            render_args=RenderArgs(
                use_alt_color_scheme=True,
            ),
            max_allowed_tokens_full_ascii_grid=300,
            spreadsheet_ascii_show_diff_if_concise=True,
        ),
        use_moderate_long_run_dots_in_system=True,
        force_reasoning_labeled_items_spreadsheet_ascii=tuple(
            reasoning_labeled_items_full_spreadsheet_alt_color_concise_diff
        ),
        force_reasoning_labeled_items_change_prompt=tuple(
            reasoning_labeled_change_prompt_alt_color_another_alt_prompt
        ),
        use_spreadsheet_if_eq_size_and_change_prompt_otherwise=True,
    ),
]


async def run_all_alt():
    # n = 1024
    # n_by_key = {"use_spreadsheet_or_change_concise_diff": 2048}

    n = 256
    # n = 128
    n_by_key = None

    out: list[tuple[str, PromptArgs, float, Optional[list[str]]]] = (
        await tqdm_asyncio.gather(
            *[
                run_on_input_with_name_alt(
                    name,
                    t=0.95,
                    n=n if n_by_key is None else n_by_key.get(prompt_args.name, n),
                    prompt_args=prompt_args,
                    max_n_per_round=48,  # avoid openai errors
                    max_n_map_if_greater=[
                        (23_000, 32),
                        (32_000, 24),
                        (38_000, 16),
                        (45_000, 8),
                    ],
                    # fail_at_prompt_len=(
                    #     80_000
                    #     if "spreadsheet" in prompt_args.name and is_eq_size_item(name)
                    #     else None
                    # ),  # idk
                    # fail_if_image_too_big_thresh=(
                    #     3000
                    #     if "spreadsheet" in prompt_args.name and is_eq_size_item(name)
                    #     else None
                    # ),
                    dry_run=False,
                )
                for prompt_args in prompt_settings
                for name in names_alt
            ]
        )
    )

    items: dict[PromptArgs, dict[str, list[str]]] = defaultdict(dict)

    for name, settings, t, completions in out:
        if completions is None:
            print(f"Failed for {name=} with {settings.name=}")
            items[settings][name] = []
            continue
        items[settings][name] = completions

    return {
        settings.name: {
            "items": vals,
            "settings": settings,
        }
        for settings, vals in items.items()
    }


# %%

# %%

out_from_all_alt = asyncio.run(run_all_alt())

# %%

# Consider running this before proceeding due to potential for OOM!!!
# subprocess.run(["redis-cli", "-p", os.environ["REDIS_READER_PORT"], "SAVE"])

# %%

list(out_from_all_alt.values())[0]["items"].keys()

# %%

eval_out_all_alt = asyncio.run(
    evaluate_funcs_with_timeout_cache(
        [
            KeyNameS(
                key=k,
                name=name,
                s=s,
                s_idx=s_idx,
            )
            for k, vs in out_from_all_alt.items()
            for name, s_vals in vs["items"].items()
            for s_idx, s in enumerate(s_vals)
        ],
        timeout=5.0,
    )
)


# %%


def process_eval_out_w_reasoning_and_outputs(
    eval_out: list[RunItem],
    by_key_w_reasoning_and_outputs: dict[
        str, dict[str, list[tuple[list[tuple[str, RunOutput]], str]]]
    ],
) -> dict[str, dict[str, list[list[tuple[str, RunOutput]]]]]:

    eval_out_dict_pre: dict[str, dict[str, list[tuple[str, int, RunOutput]]]] = (
        defaultdict(lambda: defaultdict(list))
    )

    for item in eval_out:
        eval_out_dict_pre[item.key_name_s.key][item.key_name_s.name].append(
            (
                item.key_name_s.s,
                item.key_name_s.s_idx,
                item.run_output,
            )
        )

    for k, vs in eval_out_dict_pre.items():
        for name, s_vals in vs.items():
            assert [x[1] for x in s_vals] == list(range(len(s_vals)))

            for s, s_idx, _ in s_vals:
                assert by_key_w_reasoning_and_outputs[k][name][s_idx][1] == s

    eval_out_dict = {
        k: {
            name: [
                by_key_w_reasoning_and_outputs[k][name][s_idx][0] + [(s, output)]
                for s, s_idx, output in s_vals
            ]
            for name, s_vals in vs.items()
        }
        for k, vs in eval_out_dict_pre.items()
    }

    return eval_out_dict


def process_eval_out(
    eval_out: list[RunItem],
) -> dict[str, dict[str, list[tuple[RunOutput, str]]]]:

    eval_out_dict_pre: dict[str, dict[str, list[tuple[str, int, RunOutput]]]] = (
        defaultdict(lambda: defaultdict(list))
    )

    for item in eval_out:
        eval_out_dict_pre[item.key_name_s.key][item.key_name_s.name].append(
            (
                item.key_name_s.s,
                item.key_name_s.s_idx,
                item.run_output,
            )
        )

    for k, vs in eval_out_dict_pre.items():
        for name, s_vals in vs.items():
            assert [x[1] for x in s_vals] == list(range(len(s_vals)))

    eval_out_dict = {
        k: {
            name: [(output, s) for s, _, output in s_vals]
            for name, s_vals in vs.items()
        }
        for k, vs in eval_out_dict_pre.items()
    }

    return eval_out_dict


# %%

eval_out_dict_main = process_eval_out(eval_out_all_alt)

# %%

# prefix = "train_" if use_train_set else "test_"

# with open(f"{prefix}eval_out_dict.json", "w") as f:
#     f.write(json_converter.dumps(eval_out_dict_main))

# # %%

# with open(f"{prefix}eval_out_dict.json", "r") as f:
#     loaded_eval_out_dict_main = json_converter.loads(
#         f.read(), dict[str, dict[str, list[tuple[RunOutput, str]]]]
#     )

# assert loaded_eval_out_dict_main == eval_out_dict_main


# %%

T = TypeVar("T")

# %%


def map_item_eval(
    eval_out_dict: dict[str, dict[str, list[tuple[RunOutput, str]]]],
    x: Callable[[RunOutput], T],
) -> dict[str, dict[str, list[T]]]:
    return {
        k: {name: [x(output) for output, _ in s_vals] for name, s_vals in vs.items()}
        for k, vs in eval_out_dict.items()
    }


def map_all_items_eval(
    eval_out_dict: dict[str, dict[str, list[tuple[RunOutput, str]]]],
    x: Callable[[list[RunOutput]], T],
) -> dict[str, dict[str, T]]:
    return {
        k: {name: x([output for output, _ in s_vals]) for name, s_vals in vs.items()}
        for k, vs in eval_out_dict.items()
    }


# %%


def get_filt_has_all_train_corr(
    eval_out_dict: dict[str, dict[str, list[tuple[RunOutput, str]]]]
) -> dict[str, dict[str, list[tuple[RunOutput, str]]]]:
    return {
        k: {
            name: [(l, y) for l, y in v if l.all_train_correct()]
            for name, v in vs.items()
        }
        for k, vs in eval_out_dict.items()
    }


# %% [markdown]

# # fancy shit, not most of action

# %%

total_all_train_corr_by_name: dict[str, int] = defaultdict(int)

filt_has_all_train_corr = get_filt_has_all_train_corr(
    {
        k: {name: x.get(name, []) for name in names_alt}
        for k, x in eval_out_dict_main.items()
    }
)

for k, vs in filt_has_all_train_corr.items():
    for name, v in vs.items():
        total_all_train_corr_by_name[name] += len(v)


names_all_issues = [
    name
    for name, count_train_corr in total_all_train_corr_by_name.items()
    if count_train_corr <= 32
    # if count_train_corr <= 0
]
names_no_train_corr = [
    name
    for name, count_train_corr in total_all_train_corr_by_name.items()
    if count_train_corr <= 0
]

len(names_all_issues), len(names_no_train_corr)


# %%


def get_fix_prompt_args_examples_for_key_name(key: str, name: str):
    if "use_spreadsheet_or_change" in key:
        if is_eq_size_item(name):
            prompt_args = PromptArgs(
                name="use_spreadsheet_or_change_prompt",
                display_args=DisplayArgs(
                    spreadsheet_ascii=True,
                    spreadsheet_ascii_full=True,
                    render_args=RenderArgs(
                        use_alt_color_scheme=True,
                    ),
                    max_allowed_tokens_full_ascii_grid=300,
                    spreadsheet_ascii_show_diff_if_concise=True,
                ),
                use_moderate_long_run_dots_in_system=True,
                use_spreadsheet_if_eq_size_and_change_prompt_otherwise=False,
            )

            kwargs = dict(use_output_diff=True, use_if_fix_fail_line=False)
            return (
                prompt_args,
                code_repair_reasoning_examples_w_outputs_spreadsheet_alt_color,
                kwargs,
            )
        else:
            prompt_args = PromptArgs(
                name="alt_color_change",
                display_args=DisplayArgs(
                    render_args=RenderArgs(
                        use_alt_color_scheme=True,
                    ),
                ),
            )

            kwargs = dict(use_output_diff=True, use_if_fix_fail_line=False)

            return (
                prompt_args,
                code_repair_reasoning_examples_w_outputs_change_alt_color,
                kwargs,
            )
    elif key == "default":
        kwargs = dict(use_output_diff=True, use_if_fix_fail_line=False)
        return (
            PromptArgs(
                name="default", display_args=DisplayArgs(), legacy_color_to_index=True
            ),
            code_repair_reasoning_examples_w_outputs,
            kwargs,
        )
    else:
        assert False


# %%


def can_include_for_fix(run_output: RunOutput, name: str) -> bool:
    if any(
        x is None or not is_valid(x)
        for x in run_output.train_results + run_output.test_results
    ):
        return False

    if run_output.all_train_correct():
        return False

    train_data = out_data_by_name_d[name]["train"]
    if is_eq_size_item(name):
        assert len(run_output.train_results) == len(train_data)
        if any(
            np.array(x).shape != np.array(data["output"]).shape
            for x, data in zip(run_output.train_results, train_data)
        ):
            return False

        # made things worse on the sets I tested for some reason???
        # Super strange to do repair with output == input... (Like why would this be a better starting point...)
        # Maybe something is going on like:
        # - If the best one is the same as the input, then ~nothing so far has any signal
        # - If ~nothing has signal, then just starting from blank slate-ish case with some already done reasoning is better than no signal with messy diff.
        #   (Same as taking normal samples maybe. So no edge on doing repair relative to normal samples in this case probably. (but there is an edge in general).)
        # if all(
        #     x == data["input"] for x, data in zip(run_output.train_results, train_data)
        # ):
        #     return False

        return True

    else:
        return True


async def run_all_fix_items():
    # Concentration of compute is good here.
    # Do exponential decay with 0.75x each time
    # (1/2 also worked well.)

    # target_total_n = 128
    # # target_total_n = 3072

    # multiplier = 3 / 4

    # initial_val = target_total_n * (1 - multiplier)

    # def round_to_nearest_32(x: float) -> int:
    #     return int(32 * round(x / 32))

    # ns: list[int] = []
    # for i in range(1000):
    #     here = initial_val * multiplier**i
    #     round_nearest = round_to_nearest_32(here)
    #     if round_nearest == 0:
    #         break
    #     ns.append(round_nearest)
    ns = [64, 32]

    print(f"{sum(ns)=} {ns=}")

    items: list[
        tuple[
            int,
            str,
            PromptArgs,
            list[tuple[str, list[tuple[str, RunOutput]]]],
            str,
            list[tuple[str, RunOutput]],
            dict[str, Any],
        ]
    ] = []

    for name in names_all_issues:
        # for key in eval_out_dict.keys():
        all_run_items_w_keys = sum(
            (
                [(ro, s, key) for ro, s in v[name] if can_include_for_fix(ro, name)]
                for key, v in eval_out_dict_main.items()
            ),
            [],
        )

        train_data = out_data_by_name_d[name]["train"]
        for idx_i, idx in enumerate(
            select_best_k_items_in_terms_of_distance(
                [l.train_results for l, _, _ in all_run_items_w_keys],
                [x["output"] for x in train_data],
                k=len(ns),
                # farther_from=[x["input"] for x in train_data], # just makes things worse atm
                #
                # this isn't much better than 0.0 AFAICT.
                # This way of setting it is scale invariant (up to 0.25) because select_best_k normalizes by dividing by k.
                weight_on_further_from_best_so_far=min(
                    0.01 * len(ns), 0.25
                ),  # Maybe should also depend on n.
                extra_scores=[
                    (
                        # -15 and -5 are just from eyeballing. Corresponds to some number of ranks up. Just a priori reasoning (and looking at distribution of scores), not optimized.
                        -15
                        if "use_spreadsheet_or_change_concise_diff" == key
                        and is_eq_size_item(name)
                        else (-5 if "spreadsheet" in key else 0)
                    )
                    for _, _, key in all_run_items_w_keys
                ],
                print_scores=False,
            )
        ):
            run_output, reasoning, key = all_run_items_w_keys[idx]

            # print(f"{idx_i=} {idx=} {name=} {key=}")
            # os.makedirs("issues_outs", exist_ok=True)
            # with open(
            #     f"issues_outs/reasoning_{name}_idx_i_{idx_i}_idx_{idx}.txt", "w"
            # ) as f:
            #     f.write(reasoning)

            assert not any(x == [] for x in run_output.train_results)

            assert not any(x == [] for x in run_output.test_results)

            for x in run_output.train_results:
                arr = np.array(x)
                assert arr.ndim == 2
            for x in run_output.test_results:
                arr = np.array(x)
                assert arr.ndim == 2

            this_n = ns[idx_i]

            prompt_args, code_repair_examples, fix_kwargs = (
                get_fix_prompt_args_examples_for_key_name(key, name)
            )

            items.append(
                (
                    this_n,
                    name,
                    prompt_args,
                    code_repair_examples,
                    key,
                    [(reasoning, run_output)],
                    fix_kwargs,
                )
            )

    ret: list[
        tuple[str, list[tuple[str, RunOutput]], PromptArgs, Optional[list[str]]]
    ] = await tqdm_asyncio.gather(
        *(
            fix_on_input(
                name,
                all_reasoning_and_outputs=all_reasoning_and_outputs,
                example_all_reasoning_and_outputs=examples,
                t=0.95,
                n=this_n,
                max_n_per_round=48,
                max_n_map_if_greater=[
                    (25_000, 32),
                    (32_000, 24),
                    (38_000, 16),
                    (45_000, 8),
                ],
                args=prompt_args,
                use_explicit_start=False,  # TODO: test more carefully later, but probably doesn't matter...
                # do_print_prompt=True,
                **fix_kwargs,
            )
            for this_n, name, prompt_args, examples, key, all_reasoning_and_outputs, fix_kwargs in items
        )
    )

    out: dict[str, list[str]] = defaultdict(list)
    out_w_reasoning_and_outputs: dict[
        str, list[tuple[list[tuple[str, RunOutput]], str]]
    ] = defaultdict(list)

    for name, all_reasoning_and_outputs, _, reasoning_list in ret:
        if reasoning_list is None:
            print(f"Failed for {name=}")
            continue

        out[name].extend(reasoning_list)
        out_w_reasoning_and_outputs[name].extend(
            [(all_reasoning_and_outputs, s) for s in reasoning_list]
        )

    return out, out_w_reasoning_and_outputs


# %%

fix_items_out, fix_items_with_w_reasoning_and_outputs = asyncio.run(run_all_fix_items())

# %%

eval_out_after_fix = asyncio.run(
    evaluate_funcs_with_timeout_cache(
        [
            KeyNameS(
                key="post_fix",
                name=name,
                s=s,
                s_idx=s_idx,
            )
            for name, s_vals in fix_items_out.items()
            for s_idx, s in enumerate(s_vals)
        ],
        timeout=5.0,
    )
)

# %%

eval_out_after_fix_dict = process_eval_out(eval_out_after_fix)

# %%

ALLOWED_ATTEMPTS = 2

# %%

all_fully_merged_eval: dict[str, list[tuple[RunOutput, str]]] = defaultdict(list)

for k, vs in eval_out_dict_main.items():
    for name, v in vs.items():
        all_fully_merged_eval[name].extend(v)

for _, vs in eval_out_after_fix_dict.items():
    for name, v in vs.items():
        all_fully_merged_eval[name].extend(v)

submission_dict = make_submission_dict(
    all_fully_merged_eval,
    n_to_submit=ALLOWED_ATTEMPTS,
)

# %%

with open("submission.json", "w") as file:
    json.dump(submission_dict, file, indent=4)

# %%

with open("submission.json", "r") as file:
    loaded_sub = json.load(file)

expected_sub_file = "arc-agi_evaluation_solutions.json"

# %%

# loaded_sub["60c09cac"]

# %%

with open(expected_sub_file, "r") as file:
    expected_sub = json.load(file)

score_by, overall_score = score_submission_dict(
    loaded_sub, expected_sub, allowed_attempts=ALLOWED_ATTEMPTS
)

print(f"{overall_score=}")

# %%


def score_this(merged_eval):
    submission_dict = make_submission_dict(
        merged_eval,
        n_to_submit=ALLOWED_ATTEMPTS,
    )
    _, overall_score = score_submission_dict(
        submission_dict, expected_sub, allowed_attempts=ALLOWED_ATTEMPTS
    )

    print(f"{overall_score=}")

    return overall_score


# %%


ks = np.logspace(0, 8, num=9, base=2, dtype=int)

V = TypeVar("V")


def run_for_k(
    k: int, max_chunks: int = 16, key="use_spreadsheet_or_change_concise_diff"
):
    print(f"{k=}")

    def split_into_chunks(x: list[V], k: int, max_chunks: int) -> list[list[V]]:
        assert k <= len(x)

        if k > len(x) // 2:
            out_chunks = [x[:k]]
        else:
            out_chunks = [x[i : i + k] for i in list(range(0, len(x), k))[:max_chunks]]

        for chunk in out_chunks:
            assert len(chunk) == k

        return out_chunks

    split_dict = {
        name: split_into_chunks(
            eval_out_dict_main[key].get(name, []),
            k,
            max_chunks=max_chunks,
        )
        for name in names_alt
    }

    num_splits = len(next(iter(split_dict.values())))
    for _, splits in split_dict.items():
        assert len(splits) == num_splits

    by_split_dicts = [
        {name: splits[i] for name, splits in split_dict.items()}
        for i in range(num_splits)
    ]

    return np.mean([score_this(x) for x in by_split_dicts])


# %%

perfs_by_k = [run_for_k(k, max_chunks=max(256 // k, 1)) for k in ks]
# log_incor_by_k = np.log2(1 - np.array(perfs_by_k))
lin_fit = np.polyfit(np.log2(ks[3:]), perfs_by_k[3:], 1)
# log_incor_fit = np.polyfit(np.log2(ks)[3:], log_incor_by_k[3:], 1)

# %%

import math

import matplotlib.pyplot as plt

# %%


def make_plot(
    show_fit: bool,
    use_log_incor: bool = False,
    extrapolate_fit: bool = False,
    extrapolate_fit_to: int = 23,
    show_fit_with_revision_frac: Optional[float] = None,
):
    plt.clf()

    plt.rcParams.update({"font.size": 20})

    fig, ax = plt.subplots(figsize=(24, 20))

    ks_to_use_v2 = ks

    perf_to_use = perfs_by_k
    fit_to_use = lin_fit

    if show_fit:
        ks_to_use_v2 = ks_to_use_v2[3:]
        perf_to_use = perf_to_use[3:]

    if extrapolate_fit:
        assert show_fit

        k_start = ks_to_use_v2[0]

        start_point = math.floor(math.log2(k_start)) + 1
        ks_to_use_for_fit = np.concatenate(
            [
                np.array(ks_to_use_v2),
                np.logspace(
                    start_point,
                    extrapolate_fit_to,
                    num=extrapolate_fit_to - start_point + 1,
                    base=2,
                    dtype=int,
                ),
            ]
        ).tolist()
        ks_to_use_for_fit_v2 = ks_to_use_for_fit
        ks_to_use_for_fit_v0 = ks_to_use_for_fit
        ks_to_use_tick = ks_to_use_for_fit_v2
    else:
        ks_to_use_for_fit_v2 = ks_to_use_v2
        ks_to_use_tick = ks_to_use_v2

    ax.plot(ks_to_use_v2, perf_to_use, label="V2")

    if show_fit:
        v2_fit_vals = np.polyval(fit_to_use, np.log2(ks_to_use_for_fit_v2))
        ax.plot(
            ks_to_use_for_fit_v2,
            v2_fit_vals,
            label=f"fit V2: {fit_to_use[0]:.3f}x + {fit_to_use[1]:.3f}",
        )
        if show_fit_with_revision_frac is not None:
            assert not use_log_incor
            # v2_fit_revision_vals = 1 - (
            #     (1 - v2_fit_vals) * (1 - show_fit_with_revision_frac)
            # )
            rem_revision = 1 - show_fit_with_revision_frac
            v2_fit_revision_vals = (
                rem_revision * v2_fit_vals + show_fit_with_revision_frac
            )
            ax.plot(
                ks_to_use_for_fit_v2,
                v2_fit_revision_vals,
                label=f"fit V2 w/ revision: {rem_revision * fit_to_use[0]:.3f}x + {rem_revision * fit_to_use[1] + show_fit_with_revision_frac:.3f}",
            )

    ax.set_xscale("log", base=2)

    # Customize the ticks
    ax.set_xticks(ks_to_use_tick)

    # Format the ticks to show the full number instead of the exponent
    ax.get_xaxis().set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f"{int(x)}" if x < 10_000 else f"$2^{{{int(np.log2(x))}}}$"
        )
    )

    ax.axhline(y=0.0, color="black", linestyle="-")

    ax.set_xlabel("k")
    if use_log_incor:
        ax.set_ylabel("Log top-3 incorrectness rate")
    else:
        ax.set_ylabel("Top-3 accuracy")
    if show_fit:
        plt.legend()

    return fig, ax


# %%

# make_plot(show_fit=True)
