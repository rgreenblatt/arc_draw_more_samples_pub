# %%
# import os

# # %%

# os.environ["REDIS_READER_PORT"] = "6381"
# os.environ["INPUT_JSON"] = "./arc-agi_evaluation_challenges.json"
# os.environ["INPUT_TRAIN_JSON"] = "./arc-agi_training_challenges.json"
# os.environ["INPUT_JSON_SOLUTIONS"] = (
#     "./arc-agi_evaluation_solutions.json"  # NOTE: it is allowed for this to be unset or invalid
# )
# os.environ["RUN_FULL_SAMPLES"] = "0"

# os.environ["RUN_ON_SUBSET"] = "100"  # optional integer

# # os.environ["BAN_UNCACHED_LLM_QUERIES"] = "1"

# %%

# %%

import math
from collections import defaultdict
import contextlib
import os
import json
import random
from typing import Any, Callable, Optional, TypeVar
import itertools
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
    get_alternative_system_prompt,
    get_rule_input_alt,
    is_eq_size_item,
    make_all_fix_prompt_alt,
    make_prompt_alt,
    print_prompt,
    process_prompt_args_for_name,
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
    evaluate_funcs_with_timeout_cache,
)
from arc_solve.edit_distance import is_valid, select_best_k_items_in_terms_of_distance
from arc_solve.load_data import (
    get_subset_to_run,
    out_data_by_name_d,
)
from arc_solve.reasoning_and_labels import (
    code_repair_reasoning_examples_change_alt_color_new_long_use_diff,
    code_repair_spreadsheet_alt_color_reasoning_examples_swap,
    basic_change_alt_color_variants,
    reasoning_labeled_items_full_spreadsheet_alt_color_concise_diff_variants,
    example_20_full_spreadsheet_reasoning_with_diff,
    example_20_full_spreadsheet_reasoning_with_diff_use_connected_diagonal,
    example_20_full_spreadsheet,
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
            for name, s in itertools.chain(
                *reasoning_labeled_items_full_spreadsheet_alt_color_concise_diff_variants,
                *basic_change_alt_color_variants,
            )
        ],
        timeout=5.0,
    )
)
# len(eval_out_here)
for item in eval_out_here:
    assert item.run_output is not None
    assert item.run_output.all_train_correct(), f"fail at {item.key_name_s.name}"


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

names_alt = get_subset_to_run()

# %%

# prompt_args = prompt_settings_all[0]

# name_here = names_alt[1]
# print(f"{is_eq_size_item(name_here)=}")

# prompt_args_here = process_prompt_args_for_name(name_here, prompt_args)

# this_prompt = list(make_prompt_alt(prompt_args_here))
# this_prompt.append(
#     {
#         "role": "user",
#         "content": get_rule_input_alt(
#             name_here,
#             display_args=prompt_args_here.display_args,
#             shuffle_example_order_with_permutation_index=prompt_args_here.shuffle_example_order_with_permutation_index,
#             use_multi_part_transformation_rule_hint=False,
#         ),
#     }
# )

# with open("out_prompt.txt", "w") as f:
#     with contextlib.redirect_stdout(f):
#         print_prompt(this_prompt, show_images=True)

# %%

# upd = make_all_fix_prompt_alt(
#     code_repair_reasoning_examples_w_outputs_spreadsheet_alt_color,
#     args=PromptArgs(
#         name="use_spreadsheet_or_change_prompt",
#         display_args=DisplayArgs(
#             spreadsheet_ascii=True,
#             spreadsheet_ascii_full=True,
#             render_args=RenderArgs(
#                 use_alt_color_scheme=True,
#             ),
#             max_allowed_tokens_full_ascii_grid=300,
#             spreadsheet_ascii_show_diff_if_concise=True,
#             connected_include_diagonals=False,
#         ),
#         use_spreadsheet_if_eq_size_and_change_prompt_otherwise=False,
#         shuffle_example_order_with_permutation_index=None,
#         just_reasoning_additional_info_in_system=True,
#         system_use_resolve_ambiguity=False,
#         system_use_multi_part_transformation_rule_hint=False,
#         system_use_explain_connected=False,
#     ),
#     use_fix_reasoning_tags=True,
#     use_typical_issue_text=True,
# )
# print_prompt(upd, show_images=True)

# %%


def concise_diff_replace_with_connected_use_diagonal(
    x: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for ex, reasoning in x:
        if ex == example_20_full_spreadsheet:
            assert reasoning == example_20_full_spreadsheet_reasoning_with_diff
            reasoning = (
                example_20_full_spreadsheet_reasoning_with_diff_use_connected_diagonal
            )
        out.append((ex, reasoning))

    return out


# %%


def make_prompt_args(variant_idx: int) -> PromptArgs:
    connected_include_diagonals = ((variant_idx + 1) % 3) == 0
    use_multi_part_hint = (variant_idx % 2) == 0
    use_explain_connected = ((variant_idx + 1) % 5) == 0

    shuffle_perm = None if variant_idx == 0 else variant_idx - 1

    ascii_reasoning_labeled_items = (
        reasoning_labeled_items_full_spreadsheet_alt_color_concise_diff_variants[
            variant_idx
        ]
    )

    if connected_include_diagonals:
        ascii_reasoning_labeled_items = (
            concise_diff_replace_with_connected_use_diagonal(
                ascii_reasoning_labeled_items
            )
        )

    print(
        f"{variant_idx=} {connected_include_diagonals=} {use_multi_part_hint=} {use_explain_connected=} {shuffle_perm=}"
    )

    return PromptArgs(
        name=f"variant_{variant_idx}",
        display_args=DisplayArgs(
            spreadsheet_ascii=True,
            spreadsheet_ascii_full=True,
            render_args=RenderArgs(
                use_alt_color_scheme=True,
            ),
            max_allowed_tokens_full_ascii_grid=300,
            spreadsheet_ascii_show_diff_if_concise=True,
            connected_include_diagonals=connected_include_diagonals,
        ),
        force_reasoning_labeled_items_spreadsheet_ascii=tuple(
            ascii_reasoning_labeled_items
        ),
        force_reasoning_labeled_items_change_prompt=tuple(
            basic_change_alt_color_variants[variant_idx]
        ),
        use_spreadsheet_if_eq_size_and_change_prompt_otherwise=True,
        shuffle_example_order_with_permutation_index=shuffle_perm,
        # arbitrary_diff=None if variant_idx == 0 else variant_idx - 1, # not needed because we vary prompt examples
        just_reasoning_additional_info_in_system=True,
        system_use_resolve_ambiguity=True,
        system_use_multi_part_transformation_rule_hint=use_multi_part_hint,
        system_use_explain_connected=use_explain_connected,
    )


prompt_settings_all = [make_prompt_args(variant_idx) for variant_idx in range(8)]

name_to_prompt_settings = {x.name: x for x in prompt_settings_all}

assert os.environ["RUN_FULL_SAMPLES"] in {"0", "1"}, "invalid RUN_FULL_SAMPLES"

is_small_run = os.environ["RUN_FULL_SAMPLES"] == "0"


total_n = (32 * 6) if is_small_run else (96 * 8)
n_per = 32 if is_small_run else 96
n_per_by_key = None  # not currently supported


async def run_all_alt():
    assert n_per_by_key is None
    assert (total_n % n_per) == 0
    count = total_n // n_per
    print(f"{count=}")

    settings_here = prompt_settings_all[:count]

    out: list[tuple[str, PromptArgs, float, Optional[list[str]]]] = (
        await tqdm_asyncio.gather(
            *[
                run_on_input_with_name_alt(
                    name,
                    t=0.95,
                    n=(
                        n_per
                        if n_per_by_key is None
                        else n_per_by_key.get(prompt_args.name, n_per)
                    ),
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
                for prompt_args in settings_here
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

out_from_all_alt = asyncio.run(run_all_alt())

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
    if count_train_corr <= (2 if is_small_run else 4)
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
    prompt_settings = name_to_prompt_settings[key]
    assert prompt_settings.display_args.spreadsheet_ascii
    assert prompt_settings.display_args.spreadsheet_ascii_full
    assert prompt_settings.display_args.render_args.use_alt_color_scheme
    assert prompt_settings.display_args.spreadsheet_ascii_show_diff_if_concise
    assert prompt_settings.use_spreadsheet_if_eq_size_and_change_prompt_otherwise

    assert prompt_settings.just_reasoning_additional_info_in_system
    assert prompt_settings.system_use_resolve_ambiguity
    assert not prompt_settings.use_multi_part_transformation_rule_hint_on_user_call

    which_shuffle = prompt_settings.shuffle_example_order_with_permutation_index
    # print(f"{which_shuffle=} {key=}")

    kwargs = dict(
        use_output_diff=True,
        use_if_fix_fail_line=False,
        use_typical_issue_text=not prompt_settings.system_use_multi_part_transformation_rule_hint, # If less focused on rule, emphasize rule details more now.
    )

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
                connected_include_diagonals=prompt_settings.display_args.connected_include_diagonals,
            ),
            use_spreadsheet_if_eq_size_and_change_prompt_otherwise=False,
            shuffle_example_order_with_permutation_index=which_shuffle,
            just_reasoning_additional_info_in_system=True,
            system_use_resolve_ambiguity=False,
            system_use_multi_part_transformation_rule_hint=False,
            system_use_explain_connected=False,
        )

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
            shuffle_example_order_with_permutation_index=which_shuffle,
            just_reasoning_additional_info_in_system=True,
            system_use_resolve_ambiguity=False,
            system_use_multi_part_transformation_rule_hint=False,
        )

        return (
            prompt_args,
            code_repair_reasoning_examples_w_outputs_change_alt_color,
            kwargs,
        )


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

        # skip input == output
        if all(
            x == data["input"] for x, data in zip(run_output.train_results, train_data)
        ):
            return False

        return True

    else:
        return True


async def run_all_fix_items():
    # Concentration of compute is good here.
    # Do exponential decay with 0.75x each time
    # (1/2 also worked well.)

    if is_small_run:
        ns = [48, 32, 24]
    else:
        target_total_n = 384
        multiplier = 3 / 4

        initial_val = target_total_n * (1 - multiplier)

        def round_to_nearest_32(x: float) -> int:
            return int(32 * round(x / 32))

        ns: list[int] = []
        for i in range(1000):
            here = initial_val * multiplier**i
            round_nearest = round_to_nearest_32(here)
            if round_nearest == 0:
                break
            ns.append(round_nearest)

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
                use_fix_reasoning_tags=True,
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
    # if k != "variant_0":
    #     print(f"Skipping {k=}")
    #     continue
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

import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-4o")

total_output_tokens = sum(
    len(tokenizer.encode(reasoning))
    for _, sub_items in all_fully_merged_eval.items()
    for _, reasoning in sub_items
)
# * 1.5 is for input also.
estim_cost_total = (total_output_tokens / 1_000_000) * 15 * 1.5
print(f"{estim_cost_total=}")

# %%

with open("submission.json", "w") as file:
    json.dump(submission_dict, file, indent=4)

# %%

with open("submission.json", "r") as file:
    loaded_sub = json.load(file)


# %%

if "INPUT_JSON_SOLUTIONS" in os.environ and os.path.exists(
    os.environ["INPUT_JSON_SOLUTIONS"]
):
    expected_sub_file = os.environ["INPUT_JSON_SOLUTIONS"]

    with open(expected_sub_file, "r") as file:
        expected_sub = json.load(file)

    expected_sub = {k: expected_sub[k] for k in names_alt}

    score_by, overall_score = score_submission_dict(
        loaded_sub, expected_sub, allowed_attempts=ALLOWED_ATTEMPTS
    )

    print(f"{overall_score=}")
else:
    expected_sub = None

# %%


def score_this(merged_eval):
    submission_dict = make_submission_dict(
        merged_eval,
        n_to_submit=ALLOWED_ATTEMPTS,
    )
    assert expected_sub is not None
    _, overall_score = score_submission_dict(
        submission_dict, expected_sub, allowed_attempts=ALLOWED_ATTEMPTS
    )

    print(f"{overall_score=}")

    return overall_score


# %%

gen = np.random.default_rng(3847)

combine_by_name = {
    name: sum((x[name] for x in eval_out_dict_main.values()), []) for name in names_alt
}
lens = [len(x) for x in combine_by_name.values()]
assert set(lens) == {lens[0]}

canonical_perm = gen.permutation(lens[0])

canonical_eval_shuffle = {
    name: [values[idx] for idx in canonical_perm]
    for name, values in combine_by_name.items()
}

# %%


endpoint = math.log2(total_n)
logspace_endpoint = math.floor(endpoint)
ks = np.array(
    np.logspace(
        0, logspace_endpoint, num=logspace_endpoint + 1, base=2, dtype=int
    ).tolist()
    + ([total_n] if total_n != 2 ** logspace_endpoint else [])
)

V = TypeVar("V")


def run_for_k(k: int, max_chunks: int = 16):
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
            canonical_eval_shuffle.get(name, []),
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

fit_start = 2

if expected_sub is not None:
    perfs_by_k = [run_for_k(k, max_chunks=max(min(256, total_n) // k, 1)) for k in ks]
    # log_incor_by_k = np.log2(1 - np.array(perfs_by_k))
    lin_fit = np.polyfit(np.log2(ks[fit_start:]), perfs_by_k[fit_start:], 1)
    # log_incor_fit = np.polyfit(np.log2(ks)[fit_start:], log_incor_by_k[fit_start:], 1)

# %%

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

    ks_to_use_v3 = ks

    perf_to_use = perfs_by_k
    fit_to_use = lin_fit

    if show_fit:
        ks_to_use_v3 = ks_to_use_v3[fit_start:]
        perf_to_use = perf_to_use[fit_start:]

    if extrapolate_fit:
        assert show_fit

        k_start = ks_to_use_v3[0]

        start_point = math.floor(math.log2(k_start)) + 1
        ks_to_use_for_fit = np.concatenate(
            [
                np.array(ks_to_use_v3),
                np.logspace(
                    start_point,
                    extrapolate_fit_to,
                    num=extrapolate_fit_to - start_point + 1,
                    base=2,
                    dtype=int,
                ),
            ]
        ).tolist()
        ks_to_use_for_fit_v3 = ks_to_use_for_fit
        ks_to_use_tick = ks_to_use_for_fit_v3
    else:
        ks_to_use_for_fit_v3 = ks_to_use_v3
        ks_to_use_tick = ks_to_use_v3

    ax.plot(ks_to_use_v3, perf_to_use, label="V3")

    if show_fit:
        v3_fit_vals = np.polyval(fit_to_use, np.log2(ks_to_use_for_fit_v3))
        ax.plot(
            ks_to_use_for_fit_v3,
            v3_fit_vals,
            label=f"fit V3: {fit_to_use[0]:.3f}x + {fit_to_use[1]:.3f}",
        )
        if show_fit_with_revision_frac is not None:
            assert not use_log_incor
            # v3_fit_revision_vals = 1 - (
            #     (1 - v3_fit_vals) * (1 - show_fit_with_revision_frac)
            # )
            rem_revision = 1 - show_fit_with_revision_frac
            v3_fit_revision_vals = (
                rem_revision * v3_fit_vals + show_fit_with_revision_frac
            )
            ax.plot(
                ks_to_use_for_fit_v3,
                v3_fit_revision_vals,
                label=f"fit V3 w/ revision: {rem_revision * fit_to_use[0]:.3f}x + {rem_revision * fit_to_use[1] + show_fit_with_revision_frac:.3f}",
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
        ax.set_ylabel("Log top-2 incorrectness rate")
    else:
        ax.set_ylabel("Top-2 accuracy")
    if show_fit:
        plt.legend()

    return fig, ax


# %%

if expected_sub is not None:
    fig_out, ax_out = make_plot(show_fit=True)
    # show
    plt.show()
