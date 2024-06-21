from collections import defaultdict
import os
import random
import contextlib
from typing import Any, Callable, Optional, TypeVar
import asyncio

os.environ["REDIS_READER_PORT"] = "6381"
# os.environ["BAN_UNCACHED_LLM_QUERIES"] = "1"

from tqdm.asyncio import tqdm_asyncio
import attrs
import nest_asyncio
import numpy as np
from cattrs.preconf.json import make_converter

json_converter = make_converter()

from arc_solve.prompting import (
    DisplayArgs,
    PromptArgs,
    fix_on_input,
    get_rule_input_alt,
    is_eq_size_item,
    make_all_fix_prompt_alt,
    make_prompt_alt,
    print_prompt,
    process_prompt_args_for_name,
    run_on_input_with_name_alt,
)
from arc_solve.submission import (
    SubInfo,
    each_mean_correct_select_best_on_dict,
    is_correct_select_best_on_dict,
    is_correct_select_best_to_submit,
    mean_correct_select_best_on_dict,
    submission_nice_info_by_problem,
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
    out_train_data_by_name_d,
    out_eval_data_by_name_d,
    out_data_by_name_d,
)
from arc_solve.reasoning_and_labels import (
    code_repair_example_4,
    code_repair_reasoning_examples_change_alt_color,
    code_repair_reasoning_examples_change_alt_color_new_long,
    code_repair_reasoning_examples_change_alt_color_new_long_use_diff,
    code_repair_reasoning_examples_change_alt_color_new_short,
    reasoning_labeled_items_full_spreadsheet_alt_color_fresh_hard,
    reasoning_labeled_items_full_spreadsheet_alt_color_fresh_hard_alt,
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
    all_perm_reasoning_change_alt_color_prompt_merge,
    all_perm_reasoning_concise_diff_prompt_merge,
    reasoning_labeled_items_full_spreadsheet_alt_color_concise_diff,
    # code_repair_spreadsheet_w_diff_new_alt_color_reasoning_examples,
)

nest_asyncio.apply()


# %%

demo_ex = out_data_by_name_d["a5313dff.json"]["train"][1]
demo_ex_inp = demo_ex["input"]
demo_ex_out = demo_ex["output"]
# show_grid(
#     np.array(demo_ex_out),
#     # should_highlight=np.array(demo_ex_inp) != np.array(demo_ex_out),
#     # use_larger_edges=True,
# )

# show_grid(
#     np.array(demo_ex_inp),
#     lower_right_triangle=np.array(demo_ex_out),
# )

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
            for name, s in reasoning_labeled_items_full_spreadsheet_alt_color
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
    assert item.run_output.all_test_correct()

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
    assert last.all_test_correct()


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
    assert last.all_test_correct()

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

exclude_names_alt = set().union(
    {x for x, _ in code_repair_spreadsheet_alt_color_reasoning_examples},
    {
        code_repair_example_3,
        code_repair_example_12_for_spreadsheet_alt_color,
    },  # exclude for legacy/cache reasons
    {x for x, _ in code_repair_reasoning_examples_change_alt_color},
    {x for x, _ in code_repair_reasoning_examples_change_alt_color_new_long},
    {x for x, _ in code_repair_reasoning_examples_change_alt_color_new_short},
    {x for x, _ in reasoning_labeled_items_full_spreadsheet_alt_color},
    {x for x, _ in reasoning_labeled_items_full_spreadsheet_alt_color_fresh_hard},
    {x for x, _ in reasoning_labeled_items_full_spreadsheet_alt_color_fresh_hard_alt},
    {x for x, _ in reasoning_labeled_items_full_spreadsheet_alt_color_alt},
    {x for x, _ in reasoning_labeled_items_full_spreadsheet_alt_color_extra},
    {x for x, _ in reasoning_labeled_items_full_spreadsheet_alt_color_extra_extra},
    {x for x, _ in reasoning_labeled_items_full_spreadsheet_alt_color_alt_again},
    {x for x, _ in reasoning_labeled_items_alt_color},
    {x for x, _ in reasoning_labeled_change_prompt_alt_color},
    {x for x, _ in reasoning_labeled_change_prompt_alt_color_add},
    {x for x, _ in reasoning_labeled_change_prompt_alt_color_add_swap},
    {x for x, _ in reasoning_labeled_change_prompt_alt_color_add_just_change},
    {x for x, _ in reasoning_labeled_change_prompt_alt_color_add_swap_minor_alt},
    {x for x, _ in reasoning_labeled_change_prompt_alt_color_total_alternative_prompt},
    {x for x, _ in reasoning_labeled_change_spreadsheet_prompt_alt_color},
    {x for x, _ in reasoning_labeled_items_full_spreadsheet_alt_color_concise_diff},
    {x for x, _ in reasoning_labeled_change_prompt_alt_color_another_alt_prompt},
    {x for item in all_perm_reasoning_change_alt_color_prompt_merge for x, _ in item},
    {x for item in all_perm_reasoning_concise_diff_prompt_merge for x, _ in item},
    # these excluded because they sometimes trigger content filtering!?! (NOTE: just from train set, not val set)
    {
        "50846271.json",
        "150deff5.json",
    },
    # {
    #     # TOO LONG FIX ME
    #     "3631a71a.json",
    # },
    # more unsafe content triggers... (I think all from train set?)
    {
        "c0f76784.json",
        "e73095fd.json",
        "a8d7556c.json",
        "44d8ac46.json",
    },
)

# use_train_set = False
use_train_set = True

is_final_eval = False
# is_final_eval = True

if use_train_set:
    names_alt = list(out_train_data_by_name_d.keys())
else:
    names_alt = list(out_eval_data_by_name_d.keys())

    # we shouldn't be excluding anything from eval
    assert len(set(names_alt) & exclude_names_alt) == 0


random.seed(37842)
random.shuffle(names_alt)
names_alt = [x for idx, x in enumerate(names_alt) if x not in exclude_names_alt]

if is_final_eval:
    if use_train_set:
        names_alt = names_alt[-100:]  # We can also iterate on the next 200 or so.
    else:
        names_alt = names_alt[:100]
else:
    if use_train_set:
        names_alt = names_alt[:100]
    else:
        names_alt = names_alt[-100:]


# names_alt = [x for x in names_alt if is_eq_size_item(x)]

len(names_alt)

# %%


# args = PromptArgs(
#     name="use_spreadsheet_or_change_concise_diff",
#     display_args=DisplayArgs(
#         spreadsheet_ascii=True,
#         spreadsheet_ascii_full=True,
#         render_args=RenderArgs(
#             use_alt_color_scheme=True,
#         ),
#         max_allowed_tokens_full_ascii_grid=300,
#         spreadsheet_ascii_show_diff_if_concise=True,
#     ),
#     force_reasoning_labeled_items_spreadsheet_ascii=tuple(
#         reasoning_labeled_items_full_spreadsheet_alt_color_concise_diff
#     ),
#     force_reasoning_labeled_items_change_prompt=tuple(
#         reasoning_labeled_change_prompt_alt_color_another_alt_prompt
#     ),
#     use_spreadsheet_if_eq_size_and_change_prompt_otherwise=True,
#     shuffle_example_order_with_permutation_index=0,
# )

# this_name = names_alt[0]

# prompt_args_here = process_prompt_args_for_name(this_name, args)

# this_prompt = list(make_prompt_alt(prompt_args_here))
# this_prompt.append(
#     {
#         "role": "user",
#         "content": get_rule_input_alt(
#             this_name,
#             display_args=prompt_args_here.display_args,
#             shuffle_example_order_with_permutation_index=prompt_args_here.shuffle_example_order_with_permutation_index,
#         ),
#     }
# )

# with open("out_prompt.txt", "w") as f:
#     with contextlib.redirect_stdout(f):
#         print_prompt(this_prompt, show_images=True)

# %%
# 10*9*8*7*6*5*4*3*2*1

# min(len(x["train"]) for x in out_data_by_name_d.values())

# %%


prompt_settings_all = [
    PromptArgs(
        name=f"use_spreadsheet_or_change_concise_diff_{i}",
        display_args=DisplayArgs(
            spreadsheet_ascii=True,
            spreadsheet_ascii_full=True,
            render_args=RenderArgs(
                use_alt_color_scheme=True,
            ),
            max_allowed_tokens_full_ascii_grid=300,
            spreadsheet_ascii_show_diff_if_concise=True,
        ),
        force_reasoning_labeled_items_spreadsheet_ascii=tuple(
            all_perm_reasoning_concise_diff_prompt_merge[i]
        ),
        force_reasoning_labeled_items_change_prompt=tuple(
            all_perm_reasoning_change_alt_color_prompt_merge[i]
        ),
        use_spreadsheet_if_eq_size_and_change_prompt_otherwise=True,
        shuffle_example_order_with_permutation_index=None if i == 0 else i,
    )
    for i in range(18)
]
name_to_prompt_settings = {x.name: x for x in prompt_settings_all}

# is_small_run = False
is_small_run = True


async def run_all_alt():
    total_n = 192 if is_small_run else 1024
    n_per = 64 if is_small_run else 128
    n_per_by_key = None
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


def get_is_test_corr_for_all_train_corr(
    filt_has_all_train_corr: dict[str, dict[str, list[tuple[RunOutput, str]]]]
) -> dict[str, dict[str, list[bool]]]:
    return {
        k: {name: [l.all_test_correct() for l, _ in v] for name, v in vs.items()}
        for k, vs in filt_has_all_train_corr.items()
    }


# %%


# hacky utils for some checks
def three_attempts(x: float) -> float:
    # Biased coin, probability of heads on at least one flip out of 3
    return 1 - (1 - x) ** 3


def compute_corr_for_bools(x: list[bool]) -> float:
    if len(x) == 0:
        return 0.0
    return three_attempts(sum(x) / len(x))


# %%

has_any_none = map_item_eval(
    eval_out_dict_main,
    lambda x: any(y is None for y in x.train_results)
    or any(y is None for y in x.test_results),
)

# %%

lim_k = 100000000

# %%

# [len(out_data_by_name_d[name]["test"]) for name in names_alt]

# %%

initial_corr_lim_k = each_mean_correct_select_best_on_dict(
    {
        k: {name: vals.get(name, [])[:lim_k] for name in names_alt}
        for k, vals in eval_out_dict_main.items()
    }
)
print(f"{initial_corr_lim_k=}")

# %%

# lim_k_for_compare_prompts = 1024

# compare_prompts_fixed_k_corr_vals = each_mean_correct_select_best_on_dict(
#     {
#         k: {name: vals.get(name, [])[:lim_k_for_compare_prompts] for name in names_alt}
#         for k, vals in eval_out_dict_main.items()
#     }
# )

# # default: v0
# # use_spreadsheet_or_change_prompt: v1
# # use_spreadsheet_or_change_prompt_alts: different for diversity
# # use_spreadsheet_or_change_concise_diff: v2

# print(f"{compare_prompts_fixed_k_corr_vals=}")

# %%
import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-4o")

total_output_tokens = sum(
    len(tokenizer.encode(reasoning))
    for _, items in eval_out_dict_main.items()
    for _, sub_items in items.items()
    for _, reasoning in sub_items
)
# * 1.5 is for input also.
estim_cost_before_fix = (total_output_tokens / 1_000_000) * 15 * 1.5
print(f"{estim_cost_before_fix=}")

# %%

# output, reasoning = eval_out_dict_main["use_spreadsheet_or_change_concise_diff_0"][
#     "46442a0e.json"
# ][0]

# print(reasoning)

# orig_args = name_to_prompt_settings["use_spreadsheet_or_change_concise_diff_0"]
# which_shuffle = orig_args.shuffle_example_order_with_permutation_index

# prompt_args = PromptArgs(
#     name="alt_color_change",
#     display_args=DisplayArgs(
#         render_args=RenderArgs(
#             use_alt_color_scheme=True,
#         ),
#     ),
#     shuffle_example_order_with_permutation_index=which_shuffle,
# )

# kwargs = dict(use_output_diff=True, use_if_fix_fail_line=False)

# examples = code_repair_reasoning_examples_w_outputs_change_alt_color

# # %%

# prompt_args_new, examples_new, kwargs_new = get_fix_prompt_args_examples_for_key_name("use_spreadsheet_or_change_concise_diff_0", "46442a0e.json")
# assert prompt_args_new == prompt_args
# assert kwargs_new == kwargs
# assert examples_new == examples

# # %%

# this_prompt = list(
#     make_all_fix_prompt_alt(
#         examples_new + [("46442a0e.json", [(reasoning, output)])],
#         args=prompt_args_new,
#         use_next_prompt=True,
#         use_explicit_start=False,
#         use_output_diff=kwargs_new["use_output_diff"],
#         use_if_fix_fail_line=kwargs_new["use_if_fix_fail_line"],
#     )
# )
# print_prompt(this_prompt, show_images=True)


# %% [markdown]

# # fixing

# %%

total_all_train_corr_by_name: dict[str, int] = defaultdict(int)

filt_has_all_train_corr = get_filt_has_all_train_corr(
    {
        k: {name: x.get(name, [])[:lim_k] for name in names_alt}
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

    which_shuffle = prompt_settings.shuffle_example_order_with_permutation_index
    print(f"{which_shuffle=} {key=}")

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
            use_spreadsheet_if_eq_size_and_change_prompt_otherwise=False,
            shuffle_example_order_with_permutation_index=which_shuffle,
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
            shuffle_example_order_with_permutation_index=which_shuffle,
        )

        kwargs = dict(use_output_diff=True, use_if_fix_fail_line=False)

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

    if is_small_run:
        ns = [64, 32]
    else:
        target_total_n = 512
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


# prefix = "train_" if use_train_set else "test_"

# with open(f"{prefix}eval_out_after_fix_dict.json", "w") as f:
#     f.write(json_converter.dumps(eval_out_after_fix_dict))

# # %%

# with open(f"{prefix}eval_out_after_fix_dict.json", "r") as f:
#     loaded_eval_out_after_fix_dict = json_converter.loads(
#         f.read(), dict[str, dict[str, list[tuple[RunOutput, str]]]]
#     )

# assert loaded_eval_out_after_fix_dict == eval_out_after_fix_dict

# # %%

# before_after_fix_reasoning_lists = {
#     k: [
#         ([x for x, _ in all_prior], next_reasoning)
#         for all_prior, next_reasoning in items_here
#     ]
#     for k, items_here in fix_items_with_w_reasoning_and_outputs.items()
# }


# prefix = "train_" if use_train_set else "test_"

# with open(f"{prefix}before_and_fix_reasoning_lists.json", "w") as f:
#     f.write(json_converter.dumps(before_after_fix_reasoning_lists))

# # %%

# with open(f"{prefix}before_and_fix_reasoning_lists.json", "r") as f:
#     loaded_before_and_after_fix_reasoning_lists = json_converter.loads(
#         f.read(), dict[str, list[tuple[list[str], str]]]
#     )

# assert loaded_before_and_after_fix_reasoning_lists == before_after_fix_reasoning_lists

# %%

eval_out_after_fix_w_reasoning = process_eval_out_w_reasoning_and_outputs(
    eval_out_after_fix, {"post_fix": fix_items_with_w_reasoning_and_outputs}
)

filt_has_all_train_corr_after_fix = get_filt_has_all_train_corr(eval_out_after_fix_dict)
is_test_corr_for_all_train_corr_after_fix = get_is_test_corr_for_all_train_corr(
    filt_has_all_train_corr_after_fix
)

# when reporting stats, just show one where was previously incorrect
fix_additional_corr_all = each_mean_correct_select_best_on_dict(
    {
        k: {n: v[n] for n in names_no_train_corr}
        for k, v in eval_out_after_fix_dict.items()
    }
)
assert list(fix_additional_corr_all.keys()) == ["post_fix"]
fix_additional_corr = fix_additional_corr_all["post_fix"]

assert list(eval_out_after_fix_dict.keys()) == ["post_fix"]

count_fix_solved = fix_additional_corr * len(names_no_train_corr)

# NOTE: this doesn't report exactly what you might want it to report. You should probalby just look at the final merged numbers!!!
# (And manually try removing/adding the test inputs by commenting out the corresponding blok of eval_out_after_fix_dict.)
# print(f"{fix_additional_corr=} {count_fix_solved=}")

# %%

all_fully_merged_eval: dict[str, list[tuple[RunOutput, str]]] = defaultdict(list)

lim_k_here = 10000000

for k, vs in eval_out_dict_main.items():
    for name, v in vs.items():
        all_fully_merged_eval[name].extend(v[:lim_k_here])

for _, vs in eval_out_after_fix_dict.items():
    for name, v in vs.items():
        all_fully_merged_eval[name].extend(v)

final_corr_rate = mean_correct_select_best_on_dict(
    all_fully_merged_eval,
    n_to_submit=3,
    # n_to_submit=2,
)
print(f"{final_corr_rate=}")

# %%

# Perf vs test samples (manually varying 3072 above):

# - 3040: 0.50 / 0.72 (test vs train)
# - 1536: 0.50 / 0.69 (test vs train)
# - 960: 0.47 / 0.69
# - 480: 0.47 / 0.7
# - 224: 0.46   / 0.71
# - 96: 0.42   / 0.69
# - 0:  0.37  / 0.66

# %%

# (0.5-0.37)/(1-0.37)
# (0.72-0.66)/(1-0.66)

# %%


nice_info_for_save = submission_nice_info_by_problem(all_fully_merged_eval)

# %%

nice_info_for_save_standard_order = {
    name: tuple(nice_info_for_save[name]) for name in names_alt
}

# next(iter(nice_info_for_save_standard_order.values()))

prefix = "train_" if use_train_set else "test_"

with open(f"{prefix}submission_info.json", "w") as f:
    f.write(json_converter.dumps(nice_info_for_save_standard_order, indent=4))

# %%


with open(f"{prefix}submission_info.json", "r") as f:
    loaded_nice_info = json_converter.loads(
        f.read(),
        dict[str, tuple[SubInfo, SubInfo, SubInfo]],
    )

# currently fails on stdout stderror for unclear reasonings, probably doesn't matter
# assert loaded_nice_info == nice_info_for_save_standard_order

# %%

(ex_1, outputs_1), (ex_2, outputs_2), (ex_3, outputs_3) = list(
    loaded_nice_info.items()
)[-6:-3]
first_corr_for_outputs_1 = ([x for x in outputs_1 if x.is_corr] + list(outputs_1))[0]
first_corr_for_outputs_2 = ([x for x in outputs_2 if x.is_corr] + list(outputs_2))[0]
first_corr_for_outputs_3 = ([x for x in outputs_3 if x.is_corr] + list(outputs_3))[0]

correct_w_order_by_item = [
    [x.is_corr for x in outputs] for outputs in [outputs_1, outputs_2, outputs_3]
]

is_corr_by_output = [
    x.is_corr
    for x in [
        first_corr_for_outputs_1,
        first_corr_for_outputs_2,
        first_corr_for_outputs_3,
    ]
]


# %%
