import math
from typing import Optional, TypeVar
import random

import numpy as np
from cattrs.preconf.json import make_converter
import matplotlib.pyplot as plt

from arc_solve.submission import mean_correct_select_best_on_dict

json_converter = make_converter()

from arc_solve.run_programs import RunOutput
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
    code_repair_example_3,
    code_repair_spreadsheet_alt_color_reasoning_examples_alt_shorter,
    reasoning_labeled_change_spreadsheet_prompt_alt_color,
    reasoning_labeled_change_prompt_alt_color_add_swap_minor_alt,
    reasoning_labeled_change_prompt_alt_color_total_alternative_prompt,
    reasoning_labeled_change_prompt_alt_color_another_alt_prompt,
    reasoning_labeled_items_full_spreadsheet_alt_color_concise_diff,
    # code_repair_spreadsheet_w_diff_new_alt_color_reasoning_examples,
)

# %%

use_train_set = True
prefix = "train_" if use_train_set else "test_"


with open(f"eval_out_dicts/{prefix}eval_out_dict.json", "r") as f:
    eval_out_dict = json_converter.loads(
        f.read(), dict[str, dict[str, list[tuple[RunOutput, str]]]]
    )

# %%

# exclude_names_alt = set()
exclude_names_alt = {x for x, _ in reasoning_labeled_items}.union(
    {x for x, _ in code_repair_spreadsheet_alt_color_reasoning_examples},
    {
        code_repair_example_3,
        code_repair_example_12_for_spreadsheet_alt_color,
    },  # exclude for legacy/cache reasons
    {x for x, _ in code_repair_reasoning_examples},
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

if use_train_set:
    names_alt = list(out_train_data_by_name_d.keys())
else:
    names_alt = list(out_eval_data_by_name_d.keys())

    # we shouldn't be excluding anything from eval
    assert len(set(names_alt) & exclude_names_alt) == 0


random.seed(37842)
random.shuffle(names_alt)
names_alt = [
    x
    for idx, x in enumerate(names_alt)
    if x not in exclude_names_alt
    # and (
    #     all(
    #         (shape := np.array(it).shape)[0] <= 9
    #         and shape[1] <= 9
    #         for z in out_data_by_name_d[x]["train"]
    #         for it in [z["input"], z["output"]]
    #     )
    #     and all(
    #         (shape := np.array(z["input"]).shape)[0] <= 9
    #         and shape[1] <= 9
    #         for z in out_data_by_name_d[x]["test"]
    #     )
    # )
]

if use_train_set:
    names_alt = names_alt[-100:]
else:
    names_alt = names_alt[:100]

len(names_alt)

# %%

ks = np.logspace(0, 11, num=12, base=2, dtype=int)

V = TypeVar("V")


def run_for_k(
    k: int, max_chunks: int = 16, key="use_spreadsheet_or_change_concise_diff"
):
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
            eval_out_dict[key].get(name, []),
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

    return np.mean([mean_correct_select_best_on_dict(x) for x in by_split_dicts])


# %%

perfs_by_k = [run_for_k(k, max_chunks=max(256 // k, 1)) for k in ks]
log_incor_by_k = np.log2(1 - np.array(perfs_by_k))
lin_fit = np.polyfit(np.log2(ks[3:]), perfs_by_k[3:], 1)
log_incor_fit = np.polyfit(np.log2(ks)[3:], log_incor_by_k[3:], 1)

# %%

ks_less = ks[:-1]

perfs_by_k_default = [
    run_for_k(k, max_chunks=max(256 // k, 1), key="default") for k in ks_less
]
log_incor_by_k_default = np.log2(1 - np.array(perfs_by_k_default))
lin_fit_default = np.polyfit(np.log2(ks_less)[3:], perfs_by_k_default[3:], 1)
log_incor_fit_default = np.polyfit(np.log2(ks_less)[3:], log_incor_by_k_default[3:], 1)

# %%


def make_plot(
    show_default: bool,
    show_fit: bool,
    use_log_incor: bool = False,
    extrapolate_fit: bool = False,
    extrapolate_fit_to: int = 23,
    show_fit_with_revision_frac: Optional[float] = None,
):
    plt.clf()

    plt.rcParams.update({"font.size": 20})

    fig, ax = plt.subplots(figsize=(24, 20))

    ks_to_use_v0 = ks_less
    ks_to_use_v2 = ks

    if use_log_incor:
        perf_to_use = log_incor_by_k
        perf_default_to_use = log_incor_by_k_default
        fit_to_use = log_incor_fit
        fit_default_to_use = log_incor_fit_default
    else:
        perf_to_use = perfs_by_k
        perf_default_to_use = perfs_by_k_default
        fit_to_use = lin_fit
        fit_default_to_use = lin_fit_default

    if show_fit:
        ks_to_use_v0 = ks_to_use_v0[3:]
        ks_to_use_v2 = ks_to_use_v2[3:]
        perf_to_use = perf_to_use[3:]
        perf_default_to_use = perf_default_to_use[3:]

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
        ks_to_use_for_fit_v0 = ks_to_use_v0
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

    if show_default:
        ax.plot(ks_to_use_v0, perf_default_to_use, label="V0")

        if show_fit:
            v0_fit_vals = np.polyval(fit_default_to_use, np.log2(ks_to_use_for_fit_v0))
            ax.plot(
                ks_to_use_for_fit_v0,
                v0_fit_vals,
                label=f"fit V0: {fit_default_to_use[0]:.3f}x + {fit_default_to_use[1]:.3f}",
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
    if show_default or show_fit:
        plt.legend()

    return fig, ax


# %%

prefix = "train_" if use_train_set else "test_"

# %%

plt.clf()

fig, ax = make_plot(show_default=False, show_fit=False)

ax.set_title("Top-3 accuracy vs k (V2 prompt)")
plt.savefig(f"{prefix}top_3_accuracy_vs_k_v2.png")

# %%

plt.clf()

fig, ax = make_plot(show_default=False, show_fit=True)

ax.set_title("Top-3 accuracy vs k with fit (V2 prompt)")
plt.savefig(f"{prefix}top_3_accuracy_vs_k_v2_fit.png")

# %%

plt.clf()

fig, ax = make_plot(show_default=True, show_fit=False)

ax.set_title("Top-3 accuracy vs k (V2 vs V0 prompt)")
plt.savefig(f"{prefix}top_3_accuracy_vs_k_v2_vs_v0.png")

# %%

plt.clf()

fig, ax = make_plot(show_default=True, show_fit=True, use_log_incor=False)

ax.set_title("Top-3 accuracy vs k with fit (V2 vs V0 prompt)")
plt.savefig(f"{prefix}top_3_accuracy_vs_k_v2_vs_v0_fit.png")

# %%

plt.clf()

fig, ax = make_plot(
    show_default=True,
    show_fit=True,
    show_fit_with_revision_frac=0.18 if use_train_set else 0.2,
    extrapolate_fit=True,
    use_log_incor=False,
    extrapolate_fit_to=21,
)

# use hline to show 85% line
if use_train_set:
    ax.axhline(y=0.85, color="teal", linestyle="--", label="85% MTurk baseline")
    ax.axhline(y=0.72, color="brown", linestyle="--", label="Current performance (72%)")
else:
    ax.axhline(y=0.85, color="teal", linestyle="--", label="85% target")
    ax.axhline(
        y=0.70, color="b", linestyle="--", label="70% (Typical MTurk performance on test?)"
    )
    ax.axhline(y=0.5, color="brown", linestyle="--", label="Current performance (50%)")
ax.legend()

ax.set_title("Top-3 accuracy vs k with fits and revision")
plt.savefig(f"{prefix}top_3_accuracy_vs_k_fit_revision.png")

# %%

# - 3040: 0.50 / 0.72 (test vs train)
# - 1536: 0.50 / 0.69 (test vs train)
# - 960: 0.47 / 0.69
# - 480: 0.47 / 0.7
# - 224: 0.46   / 0.71
# - 96: 0.42   / 0.69
# - 0:  0.37  / 0.66


train_revision_samples_vs_perf = [(0, 0.66), (96, 0.69), (224, 0.71), (480, 0.7), (960, 0.69), (1536, 0.69), (3040, 0.72)]
test_revision_samples_vs_perf = [(0, 0.37), (96, 0.42), (224, 0.46), (480, 0.47), (960, 0.47), (1536, 0.5), (3040, 0.5)]

plt.clf()

plt.rcParams.update({"font.size": 20})

fig, ax = plt.subplots(figsize=(24, 20))

which = train_revision_samples_vs_perf if use_train_set else test_revision_samples_vs_perf
prefix = "train_" if use_train_set else "test_"

ks_to_use_revision_samps = [x for x, _ in which[1:]]
ax.plot(ks_to_use_revision_samps, [y for _, y in which[1:]])

ax.set_xscale("log", base=2)

# Customize the ticks
ax.set_xticks(ks_to_use_revision_samps)

# Format the ticks to show the full number instead of the exponent
ax.get_xaxis().set_major_formatter(
    plt.FuncFormatter(
        lambda x, _: f"{int(x)}" if x < 10_000 else f"$2^{{{int(np.log2(x))}}}$"
    )
)
ax.axhline(y=which[0][1], color="brown", linestyle="--", label="Without revision")

ax.set_xlabel("Revision samples")
ax.set_ylabel("Top-3 accuracy")
ax.set_title(f"Top-3 accuracy vs revision samples ({'Train' if use_train_set else 'Test'})")
plt.savefig(f"{prefix}top_3_accuracy_vs_revision_samples.png")
