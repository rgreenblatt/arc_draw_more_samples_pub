from collections import Counter, defaultdict
import math
from functools import cache
import os
import io
import contextlib
from typing import Any, Callable, Optional, TypeVar
import base64
import asyncio
from scipy.ndimage import label

import attrs
import numpy as np
import openai
import nest_asyncio
import tiktoken

from rrutils.llm_api.base_llm import ContextLengthExceeded

nest_asyncio.apply()


from arc_solve.edit_distance import get_rank_geo_mean_score
from rrutils.llm_api.llm import ModelAPI
from arc_solve.render import (
    RenderArgs,
    grid_to_base64_png_oai_content,
    color_scheme_name,
    alt_color_scheme_name,
    alt_color_scheme_consts_name,
    color_scheme_consts_name,
)
from arc_solve.run_programs import (
    KeyNameS,
    RunOutput,
    RunOutputHashable,
    StdoutStderr,
    evaluate_funcs_with_timeout_cache,
)
from arc_solve.load_data import out_data_by_name_d
from arc_solve.reasoning_and_labels import (
    reasoning_labeled_items,
    reasoning_labeled_items_alt_color,
    reasoning_labeled_items_full_spreadsheet_alt_color,
    reasoning_labeled_change_prompt_alt_color_add_swap,
    reasoning_labeled_items_ascii,
    code_repair_reasoning_examples,
)

if "ANTHROPIC_API_KEY" not in os.environ:
    key_path = os.path.expanduser("~/.anthropic_api_key")
    try:
        with open(key_path, "r") as f:
            os.environ["ANTHROPIC_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        ...

if "OPENAI_API_KEY" not in os.environ:
    openai.api_key_path = os.path.expanduser("~/.openai_api_key_alt")
    try:
        with open(openai.api_key_path, "r") as f:
            os.environ["OPENAI_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        ...

# %%

model_api = ModelAPI(anthropic_num_threads=60, openai_fraction_rate_limit=0.8)


async def call(
    chat: list[dict[str, Any]],
    t: float = 0.0,
    max_tokens: Optional[int] = 4096,  # change as needed
    n: int = 1,
    model_ids: list[str] = ["gpt-4o-2024-05-13"],
    **kwargs,
):
    if n == 0:
        return []
    return await model_api.call_cached(
        model_ids=model_ids,
        prompt=chat,
        max_tokens=max_tokens,
        t=t,
        n=n,
        stream_per_chunk_timeout=None,
        non_streaming_timeout=500,
        max_attempts_per_api_call=50,
        **kwargs,
    )


async def call_anthropic(
    chat: list[dict[str, Any]],
    t: float = 0.0,
    max_tokens: Optional[int] = 4096,  # change as needed
    n: int = 1,
    model_ids: list[str] = ["claude-3-opus-20240229"],
    **kwargs,
):
    if n == 0:
        return []

    extra_args = {}
    if chat[0]["role"] == "system":
        system = chat[0]["content"]
        chat = chat[1:]
        extra_args["system"] = system

    return await model_api.call_cached(
        model_ids=model_ids,
        prompt=chat,
        max_tokens=max_tokens,
        t=t,
        n=n,
        **kwargs,
        **extra_args,
    )


def get_alternative_system_prompt(
    skip_image: bool = False,
    skip_ascii: bool = False,
    use_diff_highlight: bool = False,
    use_diff_triangles: bool = False,
    additional_info: bool = False,
    just_reasoning_additional_info: bool = False,
    just_attributes_additional_info: bool = False,
    use_many_ascii_representations: bool = False,
    use_alt_color_scheme: bool = False,
    legacy_color_to_index: bool = False,
    disable_absolute_in_normalized_ascii: bool = False,
    long_as_you_want: bool = False,
    use_diff_rep: bool = False,
    use_moderate_long: bool = False,
    use_legacy_diff_typo_quote: bool = True,
    allow_comma_after_etc_typo_fix: bool = False,
):
    scheme = (
        alt_color_scheme_consts_name
        if use_alt_color_scheme
        else color_scheme_consts_name
    )
    if legacy_color_to_index:
        color_to_index = ", ".join(
            # really should be capitalized, but for legacy cache support...
            f"{color_val}: {name.capitalize()}"
            for color_val, name in enumerate((scheme).values())
        )
    else:
        color_to_index = ", ".join(
            # really should be capitalized, but for legacy cache support...
            f"{name}: {color_val}"
            for color_val, name in enumerate((scheme).values())
        )

    many_ascii_rep_and_image_version_of_input_line = f"""The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). These grids will be shown to you as images and in various ASCII representations. The image and the ASCII representations for each input/output contain the same information: we just show both representations for convenience and ease of understanding. Each number corresponds to a color in the image. The correspondence is as follows: {color_to_index}."""

    many_ascii_rep_and_skip_image_version_of_input_line = f"""The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). These grids will be shown to you in various ASCII representations. Each number corresponds to a color. The correspondence is as follows: {color_to_index}."""

    abs_in_normalized_ascii_desc = """\n\nFor each shape, we indicate the non-normalized location of one cell in the grid (the first cell per shape in the prior representation) using [Absolute: LOCATION]. This is to make it easy to correspond to the prior representation and to give each shape a unique identification. We only show the absolute representation for shapes with more than 2 cells to save space."""

    if disable_absolute_in_normalized_ascii:
        abs_in_normalized_ascii_desc = ""

    # TODO: there is an extra '"' here due to a typo. Can't always fix due to cache.
    legacy_diff_quote = '"' if use_legacy_diff_typo_quote else ""
    diff_rep = f"""{legacy_diff_quote}\n\n### Color changes between the input grid and the output grid

This shows the difference between an input grid and an output grid as a list of the locations where one color changes to another. For instance, if {scheme[0]} changes to {scheme[2]} at A1 A2 B7, this would be represented as "{scheme[0]} (0) to {scheme[2]} (2): A1 A2 B7".

We will use the '...' notation as described earlier when applicable."""

    if not use_diff_rep:
        diff_rep = ""

    include_moderately_long = "moderately long " if use_moderate_long else ""

    comma_after_etc = "," if allow_comma_after_etc_typo_fix else ""

    ascii_rep_desc = f"""Here are descriptions of each of the different ASCII representations we will provide:

### Color by location representation

This is a grid of elements separated by '|'. For each element, we provide the color as a number and the location (in that order). Locations are denoted like A7 or D3, where columns are denoted with A, B, C, etc.{comma_after_etc} and rows are denoted with 1, 2, 3, etc. So, D3 corresponds to the cell in the 4th column and the 3rd row. Note that rows are 1-indexed.

### Location by color representation

This is a mapping from colors to the locations at which that color occurs. We use 'XR ... YR' to denote that row R is occupied from X to Y (inclusive). For instance, 'C5 ... G5' would correspond to 'C5 D5 E5 F5 G5'. We only use this '...' notation for {include_moderately_long}contiguous runs of cells in a row. We don't use this notation for columns.

We also separate the list into connected components (shapes). Each shape/component is separated by '|'.

### Normalized shape representation (by color)

This shows the geometry of each shape/component by "normalizing" the shape: showing the shape with the coordinates shifted such that the minimum row/column of the shape is row 1 and column A. This is useful for tasks like noticing identical shapes (in different positions with different colors).

Each shape/component is separated by '|'.{abs_in_normalized_ascii_desc}{diff_rep}

Now we're done going through the descriptions of the different ASCII representations.
""".strip()

    image_version_of_input_line = f'The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). These grids will be shown to you as both images and grids of numbers (ASCII). The image and the grid of numbers for each input/output contain the same information: we just show both representations for convenience. Each number corresponds to a color in the image. The correspondence is as follows: {color_to_index}.'

    non_image_version_of_line = f'The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). These grids will be shown as grids of numbers (just ASCII).'

    pure_image_version_of_input_line = f'The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). These grids will be shown to you as images. Each color in the image corresponds to a number. The correspondence is as follows: {color_to_index}.'

    maybe_diff_highlight_line = "\n\nWhen the input and output grids have identical dimensions and share the same color in more than 60% of their cells, we will display an additional version of both the input and output grids with cells that differ highlighted using a red border. This highlighting is to help you easily identify the differences between the input and output grids."

    maybe_diff_triangles_line = "\n\nWhen the input and output grids have identical dimensions and share the same color in more than 60% of their cells, we will display an additional image which shows the input color in the upper left triangle of the cell and the output color in the lower right triangle of the cell. Correspondingly, cells which are all one color (the upper triangle and lower triangle are the same color) are cells where the input and the output grids have the same color. This visualization is to help you easily identify and understand the differences between the input and output grids."

    additional_info_line_reasoning = """You follow a particular reasoning style. You break down complex problems into smaller parts and reason through them step by step, arriving at sub-conclusions before stating an overall conclusion. This reduces the extent to which you need to do large leaps of reasoning.

You reason in substantial detail for as is necessary to determine the transformation rule."""

    no_need_conside_as_long = "\n\nYour reasoning **can be as long as necessary**! The goal of the reasoning is just to make sure you end up with a correct implementation of the transformation rule, so **there isn't any need for your reasoning to be concise**. You should do any and all reasoning that would be useful."

    if not long_as_you_want:
        no_need_conside_as_long = ""

    additional_info_line_attributes = (
        "You are creative and accomplished at solving puzzles."
    )

    additional_info_line = f"""\n\n{additional_info_line_reasoning}{no_need_conside_as_long}\n\n{additional_info_line_attributes}"""

    if just_reasoning_additional_info:
        additional_info_line = "\n\n" + additional_info_line_reasoning
        assert not just_attributes_additional_info
        assert additional_info
    elif just_attributes_additional_info:
        additional_info_line = "\n\n" + additional_info_line_attributes
        assert additional_info

    if use_many_ascii_representations:
        assert not skip_ascii
        if skip_image:
            input_line = many_ascii_rep_and_skip_image_version_of_input_line
        else:
            input_line = many_ascii_rep_and_image_version_of_input_line

        input_line += "\n\n" + ascii_rep_desc
    elif skip_image:
        assert not skip_ascii
        input_line = non_image_version_of_line
    elif skip_ascii:
        input_line = pure_image_version_of_input_line
    else:
        input_line = image_version_of_input_line

    if not use_diff_highlight:
        maybe_diff_highlight_line = ""
    else:
        assert not skip_image

    if not use_diff_triangles:
        maybe_diff_triangles_line = ""
    else:
        assert not skip_image

    if not additional_info:
        additional_info_line = ""

    alternative_system_prompt_text = f"""You will given some number of paired example inputs and outputs. The outputs were produced by applying a transformation rule to the inputs. In addition to the paired example inputs and outputs, there is also one additional input without a known output. Your task is to determine the transformation rule and implement it in code.

{input_line}{maybe_diff_highlight_line}{maybe_diff_triangles_line}

The transformation only needs to be unambiguous and applicable to the example inputs and the additional input. It doesn't need to work for all possible inputs.

You'll need to carefully reason in order to determine the transformation rule. Start your response by carefully reasoning in <reasoning></reasoning> tags. Then, implement the transformation in code.

After your reasoning write code in triple backticks (```python and then ```). You should write a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`). You should make sure that you implement a version of the transformation which works in general (it shouldn't just work for the additional input).

Don't write tests in your python code, just output the `transform` function. (It will be tested later.){additional_info_line}"""

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": alternative_system_prompt_text,
                }
            ],
        },
    ]


# %%


@attrs.frozen
class DisplayArgs:
    render_args: RenderArgs = RenderArgs()
    skip_image: bool = False
    skip_ascii: bool = False
    use_diff_highlight: bool = False
    use_diff_triangles: bool = False
    ascii_separator: str = "|"
    spreadsheet_ascii: bool = False
    spreadsheet_ascii_full: bool = False
    spreadsheet_ascii_show_diff_if_concise: bool = False
    hacky_allow_size_mismatch_input_output: bool = False
    disable_absolute_in_normalized_ascii: bool = False
    max_allowed_tokens_per_color: Optional[int] = 200
    max_allowed_tokens_full_ascii_grid: Optional[int] = None

    def __attrs_post_init__(self):
        assert not (
            self.skip_image and self.skip_ascii
        ), "can't skip both image and ascii"

        if self.use_diff_highlight:
            assert not self.skip_image, "can't use diff without image"


# %%

spreadsheet_col_labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "AA",
    "AB",
    "AC",
    "AD",
]


def spreadsheet_ascii_grid(grid: np.ndarray, separator: str = "|"):
    rows, cols = grid.shape
    assert cols <= 30
    assert rows <= 30

    cols_header_line = separator.join([" "] + spreadsheet_col_labels[:cols])
    rest = "\n".join(
        separator.join([str(i + 1)] + [str(x) for x in row])
        for i, row in enumerate(grid)
    )

    return f"{cols_header_line}\n{rest}"


def get_spreadsheet_notation_str(i, j, quote: bool = True):
    out = f"{spreadsheet_col_labels[j]}{i+1}"
    if quote:
        out = f'"{out}"'
    return out


def spreadsheet_ascii_grid_as_color_by_location(grid: np.ndarray):
    rows, cols = grid.shape
    assert cols <= 30
    assert rows <= 30

    out = "\n".join(
        "|".join(
            f"{grid[i, j]} {get_spreadsheet_notation_str(i, j,quote=False)}"
            for j in range(cols)
        )
        for i in range(rows)
    )

    return out


tokenizer = tiktoken.encoding_for_model("gpt-4o")

# [tokenizer.decode([x]) for x in tokenizer.encode("A1 ... A7")]


def get_spreadsheet_notation_support_runs(rows_cols: list[tuple[int, int]]):
    row_cols_v = np.array(sorted(rows_cols, key=lambda x: (x[0], x[1])))

    running_str = ""

    idx = 0
    while idx < len(row_cols_v):
        r, c = row_cols_v[idx]

        count_in_a_row = 0
        for checking_idx, (n_r, n_c) in enumerate(row_cols_v[idx:]):
            if n_r == r and n_c == c + checking_idx:
                count_in_a_row += 1
            else:
                break

        if count_in_a_row > 4:
            start = get_spreadsheet_notation_str(r, c, quote=False)
            c_end = c + count_in_a_row - 1

            assert np.array_equal(row_cols_v[idx + count_in_a_row - 1], (r, c_end)), (
                row_cols_v[idx + count_in_a_row - 1],
                (r, c_end),
            )

            end = get_spreadsheet_notation_str(r, c_end, quote=False)

            running_str += f" {start} ... {end}"
            idx += count_in_a_row
        else:
            running_str += " " + get_spreadsheet_notation_str(r, c, quote=False)
            idx += 1

    return running_str



def find_contiguous_shapes(grid, color):
    labeled_array, num_features = label(grid == color)
    shapes = []
    for i in range(1, num_features + 1):
        shapes.append(np.argwhere(labeled_array == i))
    return shapes


def spreadsheet_ascii_grid_by_color_contiguous(
    shapes_by_color,
    use_alt_color_scheme: bool = True,
    max_allowed_tokens_per_color: Optional[int] = None,
):
    # TODO: support alt color scheme
    out = ""
    was_color_omitted = {}
    for color in range(11):
        was_color_omitted[color] = False
        contiguous_shapes = shapes_by_color[color]
        if len(contiguous_shapes) == 0:
            continue

        color_str = "|".join(
            get_spreadsheet_notation_support_runs(list(shape))
            for shape in contiguous_shapes
        )

        if (
            max_allowed_tokens_per_color is not None
            and len(tokenizer.encode(color_str)) > max_allowed_tokens_per_color
        ):
            color_str = " [OMITTED DUE TO EXCESSIVE LENGTH]"
            was_color_omitted[color] = True

        out += (
            f"{(alt_color_scheme_name if use_alt_color_scheme else color_scheme_name)[color]} ({color}):{color_str}"
        ) + "\n"

    return out, was_color_omitted


def diff_is_concise(grid_input: np.ndarray, grid_output: np.ndarray):
    if grid_input.shape != grid_output.shape:
        return False

    differs = grid_input != grid_output
    count_differs = differs.sum()
    if count_differs > 50 and (
        count_differs > 0.35 * grid_input.size or count_differs > 150
    ):
        return False

    grid_differs_x, grid_differs_y = differs.nonzero()

    all_color_pairs = set()
    for x, y in zip(grid_differs_x.tolist(), grid_differs_y.tolist()):
        all_color_pairs.add((grid_input[x, y], grid_output[x, y]))

    if len(all_color_pairs) > 8:
        return False

    return True


def always_diff_is_concise(name: str):
    for item in out_data_by_name_d[name]["train"]:
        if not diff_is_concise(np.array(item["input"]), np.array(item["output"])):
            return False

    return True


def spreadsheet_ascii_grid_by_color_diffs(
    grid_input: np.ndarray,
    grid_output: np.ndarray,
    use_alt_color_scheme: bool = True,
    use_expected_vs_got: bool = False,
):
    assert grid_input.shape == grid_output.shape
    grid_differs_x, grid_differs_y = (grid_input != grid_output).nonzero()
    differences_by_color_pairs: dict[tuple[int, int], list[tuple[int, int]]] = (
        defaultdict(list)
    )
    for x, y in zip(grid_differs_x.tolist(), grid_differs_y.tolist()):
        differences_by_color_pairs[(grid_input[x, y], grid_output[x, y])].append(
            (int(x), int(y))
        )

    out = ""
    for (color_input, color_output), differing_locs in sorted(
        differences_by_color_pairs.items(), key=lambda x: x[0]
    ):
        color_str = get_spreadsheet_notation_support_runs(differing_locs)

        scheme = alt_color_scheme_name if use_alt_color_scheme else color_scheme_name

        if use_expected_vs_got:
            out += (
                f"Expected {scheme[color_input]} ({color_input}) but got {scheme[color_output]} ({color_output}):{color_str}"
            ) + "\n"

        else:
            out += (
                f"{scheme[color_input]} ({color_input}) to {scheme[color_output]} ({color_output}):{color_str}"
            ) + "\n"

    return out


def spreadsheet_ascii_grid_by_color_contiguous_normalized(
    shapes_by_color,
    use_alt_color_scheme: bool = True,
    omit_by_color: Optional[dict[int, bool]] = None,
    disable_absolute_in_normalized_ascii: bool = False,
):
    # TODO: support alt color scheme
    out = ""

    for color in range(11):
        contiguous_shapes = shapes_by_color[color]
        if len(contiguous_shapes) == 0:
            continue

        shape_strs: list[str] = []
        for shape in contiguous_shapes:
            min_i = min(i for i, j in shape)
            min_j = min(j for i, j in shape)
            # basic = ",".join(
            #     get_spreadsheet_notation_str(i - min_i, j - min_j, quote=False)
            #     for i, j in
            # )

            normalized = [
                (i - min_i, j - min_j)
                for i, j in sorted(shape, key=lambda x: (int(x[0]), int(x[1])))
            ]

            basic_shape_str = get_spreadsheet_notation_support_runs(normalized)

            if len(shape) > 2 and not disable_absolute_in_normalized_ascii:
                shape_str = (
                    " [Absolute: "
                    + get_spreadsheet_notation_str(
                        shape[0][0], shape[0][1], quote=False
                    )
                    + "]"
                    + basic_shape_str
                )
            else:
                shape_str = basic_shape_str

            shape_strs.append(shape_str)

        color_str = "|".join(shape_strs)

        if omit_by_color is not None and omit_by_color.get(color, False):
            color_str = " [OMITTED DUE TO EXCESSIVE LENGTH]"

        out += (
            f"{(alt_color_scheme_name if use_alt_color_scheme else color_scheme_name)[color]} ({color}):{color_str}"
        ) + "\n"

    return out


def spreadsheet_ascii_grid_by_color_contiguous_absolute_small_shapes(
    overall_rows: int,
    overall_cols: int,
    shapes_by_color,
    use_alt_color_scheme: bool = True,
    separator: str = "|",
):
    overall_out = ""
    any_ever_used = False
    for color in range(11):
        contiguous_shapes = shapes_by_color[color]
        if len(contiguous_shapes) == 0:
            continue
        this_str = f"Color: {color}\n\n"

        any_used = False
        for shape_idx, shape in enumerate(contiguous_shapes):
            min_i = min(i for i, j in shape)
            min_j = min(j for i, j in shape)

            absolute_shifted_shape = [(i - min_i, j - min_j) for i, j in shape]

            n_rows = max(i for i, j in absolute_shifted_shape) + 1
            n_cols = max(j for i, j in absolute_shifted_shape) + 1

            if (
                (n_rows > overall_rows // 2 and n_cols > overall_cols // 2)
                or n_rows * n_cols > 50
                or n_rows * n_cols == 1
            ):
                continue

            any_used = True
            any_ever_used = True

            assert n_rows <= 30
            assert n_rows <= 30

            cols_header_line = separator.join([" "] + spreadsheet_col_labels[:n_cols])

            grid_labels = np.full((n_rows, n_cols), fill_value="O", dtype=object)

            for i, j in absolute_shifted_shape:
                grid_labels[i, j] = "X"

            rest = "\n".join(
                separator.join([str(i)] + [str(x) for x in row])
                for i, row in enumerate(grid_labels)
            )

            this_str += f'"shape_{shape_idx}_with_color_{(alt_color_scheme_name if use_alt_color_scheme else color_scheme_name)[color]}_{color}":\n\n'
            this_str += f"Bounding box shape: {n_rows} by {n_cols}\n\n"

            this_str += f"{cols_header_line}\n{rest}\n\n"

            this_str += (
                f"Normalized locations: ["
                + ", ".join(
                    get_spreadsheet_notation_str(i, j)
                    for i, j in absolute_shifted_shape
                )
                + "]\n\n"
            )

        if any_used:
            overall_out += this_str

    if not any_ever_used:
        return None

    return overall_out




def ascii_grid(grid: np.ndarray, separator: str = "|", spreadsheet_ascii: bool = False):
    if spreadsheet_ascii:
        return spreadsheet_ascii_grid(grid, separator=separator)

    return "\n".join(separator.join(str(x) for x in row) for row in grid)


def display_single_grid_alt(
    item: list[list[int]],
    display_args: DisplayArgs = DisplayArgs(),
    extra_shape_text: str = "",
):
    grid = np.array(item)
    x, y = grid.shape

    shape_text = f"Shape: {x} by {y}{extra_shape_text}\n\n"

    use_header_text = display_args.spreadsheet_ascii_full

    header_text = "### " if use_header_text else ""

    if not display_args.spreadsheet_ascii_full:
        ascii_text = (
            header_text
            + f"ASCII representation:\n\n{ascii_grid(grid, separator=display_args.ascii_separator, spreadsheet_ascii=display_args.spreadsheet_ascii)}\n\n"
        )
    else:
        assert display_args.spreadsheet_ascii
        assert display_args.spreadsheet_ascii_full
        color_by_loc_rep = spreadsheet_ascii_grid_as_color_by_location(grid)
        if (
            display_args.max_allowed_tokens_full_ascii_grid is not None
            and len(tokenizer.encode(color_by_loc_rep))
            > display_args.max_allowed_tokens_full_ascii_grid
        ):
            color_by_loc_rep = "[OMITTED DUE TO EXCESSIVE LENGTH]"
        ascii_text = f"### Color by location representation\n\n{color_by_loc_rep}\n\n"

        shapes_by_color = {
            color: find_contiguous_shapes(grid, color) for color in range(11)
        }

        out_text_by_color, was_color_omitted = (
            spreadsheet_ascii_grid_by_color_contiguous(
                shapes_by_color,
                use_alt_color_scheme=display_args.render_args.use_alt_color_scheme,
                max_allowed_tokens_per_color=display_args.max_allowed_tokens_per_color,
            )
        )

        ascii_text += f"### Location by color representation\n\n{out_text_by_color}\n\n"
        normalized_by_color_contiguous = spreadsheet_ascii_grid_by_color_contiguous_normalized(
            shapes_by_color,
            use_alt_color_scheme=display_args.render_args.use_alt_color_scheme,
            omit_by_color=was_color_omitted,
            disable_absolute_in_normalized_ascii=display_args.disable_absolute_in_normalized_ascii,
        )
        ascii_text += f"""### Normalized shape representation (by color)\n\n{normalized_by_color_contiguous}\n\n"""

    if display_args.skip_image:
        assert not display_args.skip_ascii
        return [
            {
                "type": "text",
                "text": shape_text + ascii_text,
            },
        ]

    out = [
        {
            "type": "text",
            "text": shape_text
            + ("### Image representation\n\n" if use_header_text else ""),
        },
        grid_to_base64_png_oai_content(grid, render_args=display_args.render_args),
    ]

    if not display_args.skip_ascii:
        out.append(
            {
                "type": "text",
                "text": ascii_text,
            }
        )

    return out


def display_example_alt(
    item_idx: int,
    item: dict[str, list[list[int]]],
    display_args: DisplayArgs = DisplayArgs(),
):
    fmt_num = item_idx + 1
    out = [
        {
            "type": "text",
            "text": f"# Example {fmt_num}\n\n## Input {fmt_num}\n\n",
        },
        *display_single_grid_alt(
            item["input"],
            display_args=display_args,
        ),
        {"type": "text", "text": f"## Output {fmt_num}\n\n"},
        *display_single_grid_alt(
            item["output"],
            display_args=display_args,
        ),
    ]

    if (
        display_args.spreadsheet_ascii_full
        and display_args.spreadsheet_ascii_show_diff_if_concise
    ):

        inp_grid, out_grid = np.array(item["input"]), np.array(item["output"])

        if inp_grid.shape == out_grid.shape or (
            not display_args.hacky_allow_size_mismatch_input_output
        ):
            assert inp_grid.shape == out_grid.shape

            color_changes = spreadsheet_ascii_grid_by_color_diffs(
                grid_input=inp_grid,
                grid_output=out_grid,
                use_alt_color_scheme=display_args.render_args.use_alt_color_scheme,
            )

            if not diff_is_concise(np.array(item["input"]), np.array(item["output"])):
                color_changes = "[OMITTED DUE TO EXCESSIVE LENGTH]"
        else:
            # vanity asserts
            assert display_args.hacky_allow_size_mismatch_input_output
            assert inp_grid.shape != out_grid.shape

            color_changes = "[OMITTED DUE TO SHAPE MISMATCH]"

        out.append(
            {
                "type": "text",
                "text": f"### Color changes between the input grid and the output grid\n\n{color_changes}\n\n",
            }
        )

    input_grid = np.array(item["input"])
    output_grid = np.array(item["output"])

    valid_diff = input_grid.shape == output_grid.shape and (
        np.sum(input_grid == output_grid) / input_grid.size > 0.6
    )

    if display_args.use_diff_highlight and valid_diff:
        to_highlight = input_grid != output_grid
        out.extend(
            [
                {
                    "type": "text",
                    "text": f"## Input {fmt_num} with cells that differ (from Output {fmt_num}) highlighted with a red border\n\n",
                },
                grid_to_base64_png_oai_content(
                    input_grid,
                    render_args=display_args.render_args,
                    should_highlight=to_highlight,
                ),
                {
                    "type": "text",
                    "text": f"## Output {fmt_num} with cells that differ (from Input {fmt_num}) highlighted with a red border\n\n",
                },
                grid_to_base64_png_oai_content(
                    output_grid,
                    render_args=display_args.render_args,
                    should_highlight=to_highlight,
                ),
            ]
        )

    if display_args.use_diff_triangles and valid_diff:
        out.extend(
            [
                {
                    "type": "text",
                    "text": f"## The Input {fmt_num} color is in the upper left triangle and the Output {fmt_num} color is in the lower right triangle\n\n",
                },
                grid_to_base64_png_oai_content(
                    input_grid,
                    render_args=display_args.render_args,
                    lower_right_triangle=output_grid,
                ),
            ]
        )

    return out


def display_wrong_output_alt(
    item_idx: int,
    item: Optional[list[list[int]]],
    expected_output: list[list[int]],
    stdout_stderr: StdoutStderr,
    display_args: DisplayArgs = DisplayArgs(),
    use_output_diff: bool = True,
):
    expected_shape = np.array(expected_output).shape
    x_expected, y_expected = expected_shape

    fmt_num = item_idx + 1

    basic_title = f"# Example {fmt_num}\n\n## Output for Example {fmt_num} from the incorrect `transform` function (aka actual output)\n\n"

    if stdout_stderr.stdout == "" and stdout_stderr.stderr == "":
        stdout_stderr_text = " stdout and stderr were empty."
    else:
        stdout_stderr_text = f"\n\nHere are the stdout and stderr of the function for this example:\n\n<stdout>\n{stdout_stderr.stdout}\n</stdout>\n\n<stderr>{stdout_stderr.stderr}</stderr>"

    if item == expected_output:
        return [
            {
                "type": "text",
                "text": basic_title
                + f"The output matches the expected output. (It is correct.){stdout_stderr_text}\n\n",
            }
        ]

    if item is None:
        return [
            {
                "type": "text",
                "text": basic_title
                + f"There was an error when running the function on this input.{stdout_stderr_text}\n\n",
            }
        ]

    has_some_invalid = any(not (0 <= x <= 9) for row in item for x in row)

    assert not display_args.skip_image
    assert not display_args.skip_ascii

    invalid_text = "Note that the output contains some invalid values (values that are not between 0 and 9 inclusive). These invalid values are incorrect and will need to be fixed. Invalid values are displayed in white in the image representation and the actual (invalid) value is displayed in the ASCII representation.\n\n"
    if not has_some_invalid:
        invalid_text = ""

    actual_shape = np.array(item).shape

    # TODO: diff text!!!

    grid_expected = np.array(expected_output)
    grid_actual = np.array(item)

    out = [
        {
            "type": "text",
            "text": basic_title + invalid_text + stdout_stderr_text.strip() + "\n\n",
        },
        *display_single_grid_alt(
            item,
            display_args=display_args,
            extra_shape_text=(
                f" (Shape differs from expected shape. The expected shape is: {x_expected} by {y_expected}.)"
                if actual_shape != expected_shape
                else ""
            ),
        ),
        {
            "type": "text",
            "text": "## Expected Output\n\n",
        },
        *display_single_grid_alt(
            expected_output,
            display_args=display_args,
        ),
    ]

    if use_output_diff:
        if grid_expected.shape != grid_actual.shape:
            color_changes = " [OMITTED DUE TO SHAPE MISMATCH]"
        elif not diff_is_concise(grid_input=grid_expected, grid_output=grid_actual):
            color_changes = " [OMITTED DUE TO EXCESSIVE LENGTH]"
        else:
            color_changes = spreadsheet_ascii_grid_by_color_diffs(
                grid_input=grid_expected,
                grid_output=grid_actual,
                use_alt_color_scheme=display_args.render_args.use_alt_color_scheme,
                use_expected_vs_got=True,
            )

        diff_rep_for_actual_expected = f"""## Color differences between the expected output and the actual output\n\n{color_changes}\n\n"""
        out.append(
            {
                "type": "text",
                "text": diff_rep_for_actual_expected,
            },
        )

    return out


def fix_prompt(
    name: str,
    run_output: RunOutput,
    display_args: DisplayArgs = DisplayArgs(),
    attempt_num: int = 0,
    use_output_diff: bool = True,
    use_if_fix_fail_line: bool = False,
):
    attempt_str = (
        f" (This is attempt {attempt_num + 1} at fixing the code.)"
        if attempt_num > 0
        else ""
    )

    # We should maybe support diffs, but this requires some additional work.
    assert not display_args.use_diff_highlight
    assert not display_args.use_diff_triangles

    scheme = (
        alt_color_scheme_consts_name
        if display_args.render_args.use_alt_color_scheme
        else color_scheme_consts_name
    )

    # TODO: note allow_comma_after_etc_typo_fix in sys prompt
    location_denote_as_needed = " Locations are denoted like A7 or D3, where columns are denoted with A, B, C, etc., and rows are denoted with 1, 2, 3, etc. So, D3 corresponds to the cell in the 4th column and the 3rd row. Note that rows are 1-indexed."

    location_row_cont_denote_as_needed = f"We use 'XR ... YR' to denote that row R differs from X to Y (inclusive). For instance, 'C5 ... G5' would correspond to 'C5 D5 E5 F5 G5'. We only use this '...' notation for moderately long contiguous runs of cells that differ in a row. We don't use this notation for columns."

    if display_args.spreadsheet_ascii_full:
        location_denote_as_needed = ""
        location_row_cont_denote_as_needed = (
            "We will use the '...' notation as described earlier when applicable."
        )

    diff_rep = f"""Below, we show what this incorrect `transform` function outputs for each example (if the corresponding output differed from the correct output).

We also show an ASCII representation of which cells differ between the expected output and the actual output for each example on which the function is incorrect. This is shown under "## Color differences between the expected output and the actual output". This representation shows the locations where the expected output and the actual output differ in color.{location_denote_as_needed} For instance, if at locations A1 A2 B7 the expected output has {scheme[0]} but the actual output has {scheme[2]}, this would be represented as "Expected {scheme[0]} (0) but got {scheme[2]} (2): A1 A2 B7". For each pair of expected and actual colors that differ at some location(s), it shows the list of locations where the expected output has the expected color but the actual output has the actual color.

We only show this representation if the expected output and the actual output have the same shape and if this representation would be sufficiently concise (not take up too much space).

{location_row_cont_denote_as_needed}

Ok, now here are the outputs and differences for each example:"""

    non_diff_rep_outputs = "Here is what this incorrect `transform` function outputs for each example (if the corresponding output differed from the correct output). You should compare this to the expected output for the corresponding example."

    if use_output_diff:
        show_rep = diff_rep
    else:
        show_rep = non_diff_rep_outputs

    # between the  shows the difference between an input grid and an output grid as a list of the locations where one color changes to another. For instance, if {scheme[0]} changes to {scheme[2]} at A1 A2 B7, this would be represented as "{scheme[0]} (0) to {scheme[2]} (2): A1 A2 B7".

    # We will use the '...' notation as described earlier when applicable."""

    prompt = f"""The `transform` function you implemented failed on at least one of the examples you were provided.{attempt_str} Your task is to determine what this issue is and then fix the code. The issue could be a bug in the code and/or an issue with your previous understanding of the transformation rule.

You'll need to carefully reason to determine the issue and to determine how to fix the code. Start your response by doing this reasoning in <reasoning></reasoning> tags. Then, implement the fixed transformation in code.

{show_rep}""".strip()

    exs = out_data_by_name_d[name]["train"]

    assert len(exs) == len(run_output.train_results)

    transform_outputs = sum(
        [
            display_wrong_output_alt(
                i,
                item=actual_output,
                expected_output=ex["output"],
                stdout_stderr=stdout_stderr,
                display_args=display_args,
                use_output_diff=use_output_diff,
            )
            for i, (actual_output, ex, stdout_stderr) in enumerate(
                zip(
                    run_output.train_results,
                    exs,
                    run_output.train_stdout_stderr,
                )
            )
        ],
        [],
    )

    if_fix_fail_line = "\n\nIf your attempted fix fails, you'll be called again (in the same way) to continue debugging. So, if print statements would help you debug, you can include them in your code."

    if not use_if_fix_fail_line:
        if_fix_fail_line = ""

    after_examples_prompt = f"""Ok, that is all of the actual and expected outputs.

Recall that you should start by reasoning to determine what the issue is in <reasoning></reasoning> tags. Also recall that the problem could be a bug in the code and/or an issue with your previous understanding of the transformation rule.

Once you are done reasoning, rewrite the code to fix the issue. Return the code in triple backticks (```python and then ```).{if_fix_fail_line}""".strip()

    return [
        {
            "type": "text",
            "text": prompt + "\n\n",
        },
        *transform_outputs,
        {
            "type": "text",
            "text": after_examples_prompt,
        },
    ]


def get_rule_input_alt(name: str, display_args: DisplayArgs = DisplayArgs()):
    exs = out_data_by_name_d[name]["train"]

    start = []

    out = start + sum(
        [
            display_example_alt(i, ex, display_args=display_args)
            for i, ex in enumerate(exs)
        ],
        [],
    )

    test = out_data_by_name_d[name]["test"]

    for test_idx, t in enumerate(test):
        test_idx_str = "" if len(test) == 1 else f" ({test_idx + 1})"
        out.extend(
            [
                {
                    "type": "text",
                    "text": f"# Additional input{test_idx_str}\n\n",
                },
                *display_single_grid_alt(t["input"], display_args=display_args),
            ]
        )

    return out


# %%


@attrs.frozen
class PromptArgs:
    name: str = "default"
    display_args: DisplayArgs = DisplayArgs()
    additional_info_in_system: bool = True
    use_spreadsheet_if_eq_size_and_change_prompt_otherwise: bool = False
    just_reasoning_additional_info_in_system: bool = False
    just_attributes_additional_info_in_system: bool = False
    force_reasoning_labeled_items: Optional[tuple[tuple[str, str], ...]] = (
        None  # tuple for hash
    )
    force_reasoning_labeled_items_spreadsheet_ascii: Optional[
        tuple[tuple[str, str], ...]
    ] = None
    force_reasoning_labeled_items_change_prompt: Optional[
        tuple[tuple[str, str], ...]
    ] = None
    legacy_color_to_index: bool = False
    emphasize_long_in_system: bool = False
    use_moderate_long_run_dots_in_system: bool = False

    def __attrs_post_init__(self):
        if self.use_spreadsheet_if_eq_size_and_change_prompt_otherwise:
            assert not self.display_args.skip_image
            assert not self.display_args.skip_ascii
            assert self.display_args.spreadsheet_ascii
            assert self.display_args.spreadsheet_ascii_full
            assert self.display_args.render_args.use_alt_color_scheme


# %%


@cache
def make_prompt_alt(
    args: PromptArgs = PromptArgs(),
):
    assert (
        not args.use_spreadsheet_if_eq_size_and_change_prompt_otherwise
    ), "this needs to be handled at an earlier stage!"
    basic_prompt = list(
        get_alternative_system_prompt(
            skip_image=args.display_args.skip_image,
            skip_ascii=args.display_args.skip_ascii,
            use_diff_highlight=args.display_args.use_diff_highlight,
            use_diff_triangles=args.display_args.use_diff_triangles,
            additional_info=args.additional_info_in_system,
            just_reasoning_additional_info=args.just_reasoning_additional_info_in_system,
            just_attributes_additional_info=args.just_attributes_additional_info_in_system,
            use_many_ascii_representations=args.display_args.spreadsheet_ascii_full,
            use_alt_color_scheme=args.display_args.render_args.use_alt_color_scheme,
            legacy_color_to_index=args.legacy_color_to_index,
            disable_absolute_in_normalized_ascii=args.display_args.disable_absolute_in_normalized_ascii,
            long_as_you_want=args.emphasize_long_in_system,
            use_diff_rep=args.display_args.spreadsheet_ascii_show_diff_if_concise,
            use_moderate_long=args.use_moderate_long_run_dots_in_system,
        )
    )

    if args.force_reasoning_labeled_items is not None:
        reasoning_labeled_items_here = list(args.force_reasoning_labeled_items)
    elif (
        args.display_args.render_args.use_alt_color_scheme
        and args.display_args.spreadsheet_ascii_full
    ):
        reasoning_labeled_items_here = (
            reasoning_labeled_items_full_spreadsheet_alt_color
        )
    elif args.display_args.render_args.use_alt_color_scheme:
        reasoning_labeled_items_here = reasoning_labeled_items_alt_color
    else:
        assert not args.display_args.render_args.use_alt_color_scheme
        assert not args.display_args.spreadsheet_ascii_full
        reasoning_labeled_items_here = (
            reasoning_labeled_items
            if not args.display_args.skip_image
            else reasoning_labeled_items_ascii
        )

    for name, reasoning in reasoning_labeled_items_here:
        basic_prompt.append(
            {
                "role": "user",
                "content": get_rule_input_alt(
                    name,
                    display_args=args.display_args,
                ),
            }
        )
        basic_prompt.append(
            {
                "role": "assistant",
                "content": reasoning,
            }
        )

    return basic_prompt


@cache
def make_fix_prompt_item(
    name: str,
    all_reasoning_and_outputs: tuple[tuple[str, RunOutputHashable], ...],
    display_args: DisplayArgs = DisplayArgs(),
    use_next_prompt: bool = False,
    use_explicit_start: bool = False,
    use_output_diff: bool = True,
    use_if_fix_fail_line: bool = False,
):
    return make_fix_prompt_item_uncache(
        name=name,
        all_reasoning_and_outputs=tuple(
            (reasoning, RunOutput.from_hashable(run_output))
            for reasoning, run_output in all_reasoning_and_outputs
        ),
        display_args=display_args,
        use_next_prompt=use_next_prompt,
        use_explicit_start=use_explicit_start,
        use_output_diff=use_output_diff,
        use_if_fix_fail_line=use_if_fix_fail_line,
    )


def make_fix_prompt_item_uncache(
    name: str,
    all_reasoning_and_outputs: tuple[tuple[str, RunOutput], ...],
    display_args: DisplayArgs = DisplayArgs(),
    use_next_prompt: bool = False,
    use_explicit_start: bool = False,
    use_output_diff: bool = True,
    use_if_fix_fail_line: bool = False,
):
    (initial_reasoning, initial_run_output), *all_fix_reasoning = (
        all_reasoning_and_outputs
    )

    if use_explicit_start:
        additional_for_prompt = [
            {
                "type": "text",
                "text": "Here are the paired example inputs and outputs. The outputs were produced by applying a transformation rule to the inputs and your task is to determine the transformation rule and implement it in code.\n\nStart your response by carefully reasoning in <reasoning></reasoning> tags. Then, implement the transformation in code.\n\n",
            },
        ]
    else:
        additional_for_prompt = []

    prompt = [
        {
            "role": "user",
            "content": additional_for_prompt
            + get_rule_input_alt(name, display_args=display_args),
        },
        {
            "role": "assistant",
            "content": initial_reasoning,
        },
    ]
    fix_prompt_part = [
        {
            "role": "user",
            "content": fix_prompt(
                name,
                initial_run_output,
                display_args=display_args,
                attempt_num=0,
                use_output_diff=use_output_diff,
                use_if_fix_fail_line=use_if_fix_fail_line,
            ),
        },
    ]
    if len(all_fix_reasoning) > 0 or use_next_prompt:
        prompt.extend(fix_prompt_part)
    else:
        print("WARNING: no fix reasoning provided!")

    for idx, (reasoning, run_output) in enumerate(all_fix_reasoning):
        prompt.append(
            {
                "role": "assistant",
                "content": reasoning,
            },
        )
        if idx != len(all_fix_reasoning) - 1 or use_next_prompt:
            prompt.append(
                {
                    "role": "user",
                    "content": fix_prompt(
                        name,
                        run_output,
                        display_args=display_args,
                        attempt_num=idx + 1,
                        use_output_diff=use_output_diff,
                        use_if_fix_fail_line=use_if_fix_fail_line,
                    ),
                },
            )

    return prompt


def make_all_fix_prompt_alt(
    items_all_reasoning_and_outputs: list[tuple[str, list[tuple[str, RunOutput]]]],
    args: PromptArgs = PromptArgs(),
    use_next_prompt: bool = False,
    use_explicit_start: bool = False,
    use_output_diff: bool = True,
    use_if_fix_fail_line: bool = False,
):
    prompt = list(
        get_alternative_system_prompt(
            skip_image=args.display_args.skip_image,
            skip_ascii=args.display_args.skip_ascii,
            use_diff_highlight=args.display_args.use_diff_highlight,
            use_diff_triangles=args.display_args.use_diff_triangles,
            additional_info=args.additional_info_in_system,
            just_reasoning_additional_info=args.just_reasoning_additional_info_in_system,
            just_attributes_additional_info=args.just_attributes_additional_info_in_system,
            use_many_ascii_representations=args.display_args.spreadsheet_ascii_full,
            use_alt_color_scheme=args.display_args.render_args.use_alt_color_scheme,
            legacy_color_to_index=args.legacy_color_to_index,
            disable_absolute_in_normalized_ascii=args.display_args.disable_absolute_in_normalized_ascii,
            long_as_you_want=args.emphasize_long_in_system,
            use_diff_rep=args.display_args.spreadsheet_ascii_show_diff_if_concise,
            use_moderate_long=args.use_moderate_long_run_dots_in_system,
        )
    )

    # NOTE: we would need the alternative reasoning trace!!!
    # assert not args.display_args.render_args.use_alt_color_scheme
    # assert not args.display_args.spreadsheet_ascii_full

    assert not args.display_args.skip_image, "not supported"

    for idx, (name, all_reasoning_and_outputs) in enumerate(
        items_all_reasoning_and_outputs
    ):
        is_last = idx == len(items_all_reasoning_and_outputs) - 1
        prompt.extend(
            make_fix_prompt_item(
                name=name,
                all_reasoning_and_outputs=tuple(
                    (
                        (reasoning_here, run_output.to_hashable())
                        for reasoning_here, run_output in all_reasoning_and_outputs
                    )
                ),
                display_args=args.display_args,
                use_next_prompt=use_next_prompt and is_last,
                use_explicit_start=use_explicit_start,
                use_output_diff=use_output_diff,
                use_if_fix_fail_line=use_if_fix_fail_line,
            )
        )

    return prompt


def print_prompt(x: list[dict[str, Any]], show_images: bool = False):
    img_idx = 0
    for item in x:
        print(f"Role: {item['role']}")
        if isinstance(item["content"], str):
            print(item["content"])
            print()
            continue
        for sub_item in item["content"]:
            assert isinstance(sub_item, dict)
            if "text" in sub_item:
                print(sub_item["text"])
            else:
                assert "image_url" in sub_item
                if show_images:
                    img = sub_item["image_url"]["url"].split(",")[1]
                    img = base64.b64decode(img)
                    img_path = f"test_{img_idx}.png"
                    with open(img_path, "wb") as f:
                        f.write(img)
                    print(img_path)
                    img_idx += 1
                else:
                    print("<IMAGE/>")
        print()


def prompt_join_text_content(x: list[dict[str, Any]]):
    out = []
    for item in x:
        if isinstance(item["content"], str):
            out.append(item)
            continue

        assert isinstance(item["content"], list)

        running_text = ""
        for sub_item in item["content"]:
            assert "text" in sub_item
            running_text += sub_item["text"]

        out.append(
            {
                "role": item["role"],
                "content": running_text,
            }
        )

    return out


def is_eq_size_item(name: str):
    train = out_data_by_name_d[name]["train"]

    return all(np.array(x["input"]).shape == np.array(x["output"]).shape for x in train)


def convert_use_spreadsheet_if_eq_size(
    prompt_args: PromptArgs, is_eq_size_and_get_spreadsheet: bool
):
    assert prompt_args.force_reasoning_labeled_items is None
    if is_eq_size_and_get_spreadsheet:
        prompt_args_here = attrs.evolve(
            prompt_args,
            use_spreadsheet_if_eq_size_and_change_prompt_otherwise=False,
            force_reasoning_labeled_items=prompt_args.force_reasoning_labeled_items_spreadsheet_ascii,
        )
        assert prompt_args_here.display_args.spreadsheet_ascii
        assert prompt_args_here.display_args.spreadsheet_ascii_full
        return prompt_args_here
    else:
        return attrs.evolve(
            prompt_args,
            display_args=attrs.evolve(
                prompt_args.display_args,
                spreadsheet_ascii=False,
                spreadsheet_ascii_full=False,
                spreadsheet_ascii_show_diff_if_concise=False,
            ),
            use_spreadsheet_if_eq_size_and_change_prompt_otherwise=False,
            force_reasoning_labeled_items=tuple(
                prompt_args.force_reasoning_labeled_items_change_prompt
                if prompt_args.force_reasoning_labeled_items_change_prompt is not None
                else reasoning_labeled_change_prompt_alt_color_add_swap
            ),
        )


def process_prompt_args_for_name(name: str, prompt_args: PromptArgs):
    if not prompt_args.use_spreadsheet_if_eq_size_and_change_prompt_otherwise:
        return prompt_args

    return convert_use_spreadsheet_if_eq_size(
        prompt_args, is_eq_size_and_get_spreadsheet=is_eq_size_item(name)
    )


async def run_on_input_alt(
    name: str,
    t: float = 0.0,
    n: int = 1,
    prompt_args: PromptArgs = PromptArgs(),
    max_n_per_round: int = 48,
    max_n_map_if_greater: list[tuple[int, int]] = [
        (25_000, 32),
        (40_000, 16),
        (65_000, 8),
    ],
    fail_at_prompt_len: Optional[int] = None,
    fail_if_image_too_big_thresh: Optional[int] = None,
    dry_run: bool = False,
):
    if n == 0:
        if dry_run:
            print(f"Return because n=0, {prompt_args.name=} {is_eq_size_item(name)=}")
        return []

    prompt_args_here = process_prompt_args_for_name(name, prompt_args)

    this_prompt = list(make_prompt_alt(prompt_args_here))
    this_prompt.append(
        {
            "role": "user",
            "content": get_rule_input_alt(
                name,
                display_args=prompt_args_here.display_args,
            ),
        }
    )

    all_user_input_image_size = sum(
        np.array(z).size
        for x in out_data_by_name_d[name]["train"]
        for z in [x["input"], x["output"]]
    ) + sum(np.array(x["input"]).size for x in out_data_by_name_d[name]["test"])

    if (
        fail_if_image_too_big_thresh is not None
        and all_user_input_image_size > fail_if_image_too_big_thresh
    ):
        print(f"fail {all_user_input_image_size=}")
        return None

    file = io.StringIO()
    with contextlib.redirect_stdout(file) as f:
        print_prompt(this_prompt, show_images=False)

    n_toks = len(tokenizer.encode(file.getvalue()))  # ignores images for now

    if fail_at_prompt_len is not None and n_toks > fail_at_prompt_len:
        print(f"fail {n_toks=} {name=} {prompt_args.name=}")
        return None

    orig_max_n_per_round = max_n_per_round

    for thresh, new_max_n in max_n_map_if_greater:
        if n_toks > thresh:
            assert max_n_per_round > new_max_n
            max_n_per_round = new_max_n

    if orig_max_n_per_round != max_n_per_round:
        print(
            f"Reducing {orig_max_n_per_round=} to {max_n_per_round=} ({n_toks=}, {name=}, {prompt_args.name=})"
        )

    if dry_run:
        print(f"{name=} {prompt_args.name=} {n_toks=} {n=} {is_eq_size_item(name)=}")
        return []

    try:
        out = await call(this_prompt, t=t, n=n, max_n_per_call=max_n_per_round)
    except Exception as e:
        print(f"{name=} {e=} {e.args=}")

        if isinstance(e, RuntimeError) and "unsafe content" in str(e):
            return None

        raise
    print(f"{out[0].token_usage=}")
    return out


async def run_on_input_with_name_alt(
    name: str,
    t: float = 0.0,
    n: int = 1,
    prompt_args: PromptArgs = PromptArgs(),
    max_n_per_round: int = 48,
    max_n_map_if_greater: list[tuple[int, int]] = [
        (25_000, 32),
        (40_000, 16),
        (65_000, 8),
    ],
    fail_at_prompt_len: Optional[int] = None,
    fail_if_image_too_big_thresh: Optional[int] = None,
    dry_run: bool = False,
):
    out = await run_on_input_alt(
        name,
        t=t,
        n=n,
        prompt_args=prompt_args,
        max_n_per_round=max_n_per_round,
        max_n_map_if_greater=max_n_map_if_greater,
        fail_at_prompt_len=fail_at_prompt_len,
        fail_if_image_too_big_thresh=fail_if_image_too_big_thresh,
        dry_run=dry_run,
    )

    return name, prompt_args, t, None if out is None else [x.completion for x in out]


def gpt4_o_image_test_single(
    grid, render_args: RenderArgs = RenderArgs(use_alt_color_scheme=True)
):
    assert render_args.use_alt_color_scheme

    x, y = grid.shape
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Here is a {x} by {y} grid. The grid cells might be any of the following colors: "black", "blue", "red", "green", "yellow", "grey", "pink", "orange", "purple", and "brown".

Your task is to determine the color of each grid in the image and return the corresponding color. You should return a list of lists in json format (which is the same as Python format in this case) where each inner list corresponds to a row of the grid and each element of the inner list is the color of the corresponding cell in the grid (one of the strings from above).

Just return the list of lists with no commentary.""",
                },
                grid_to_base64_png_oai_content(grid, render_args=render_args),
            ],
        }
    ]


def gpt4_o_image_test_few_shot(
    grid, example_grids, render_args: RenderArgs = RenderArgs(use_alt_color_scheme=True)
):
    out_prompt = []

    for ex in example_grids:
        out_prompt.extend(gpt4_o_image_test_single(ex, render_args=render_args))
        out_prompt.append(
            {
                "role": "assistant",
                "content": "[\n"
                + "\n".join(
                    " " * 4
                    + "["
                    + ", ".join(f'"{alt_color_scheme_name[x]}"' for x in row)
                    + "],"
                    for row in ex
                )
                + "\n]",
            }
        )

    out_prompt.extend(gpt4_o_image_test_single(grid, render_args=render_args))

    return out_prompt


async def fix_on_input(
    name: str,
    all_reasoning_and_outputs: list[tuple[str, RunOutput]],
    example_all_reasoning_and_outputs: list[tuple[str, list[tuple[str, RunOutput]]]],
    t: float = 1.0,
    n: int = 16,
    args: PromptArgs = PromptArgs(),
    do_print_prompt: bool = False,
    use_explicit_start: bool = False,
    max_n_per_round: int = 32,
    max_n_map_if_greater: list[tuple[int, int]] = [
        (25_000, 32),
        (40_000, 16),
        (65_000, 8),
    ],
    use_output_diff: bool = True,  # not back compat, but better probably
    use_if_fix_fail_line: bool = False,
):
    if n == 0:
        return name, all_reasoning_and_outputs, args, []

    this_prompt = list(
        make_all_fix_prompt_alt(
            example_all_reasoning_and_outputs + [(name, all_reasoning_and_outputs)],
            args=args,
            use_next_prompt=True,
            use_explicit_start=use_explicit_start,
            use_output_diff=use_output_diff,
            use_if_fix_fail_line=use_if_fix_fail_line,
        )
    )

    if do_print_prompt:
        print("=== PROMPT ===")
        print_prompt(this_prompt)

    file = io.StringIO()
    with contextlib.redirect_stdout(file) as f:
        print_prompt(this_prompt, show_images=False)

    prompt_text = file.getvalue()
    n_toks = len(tokenizer.encode(prompt_text))  # ignores images for now

    orig_max_n_per_round = max_n_per_round

    for thresh, new_max_n in max_n_map_if_greater:
        if n_toks > thresh:
            assert max_n_per_round > new_max_n
            max_n_per_round = new_max_n

    if orig_max_n_per_round != max_n_per_round:
        print(f"Reducing {orig_max_n_per_round=} to {max_n_per_round=} ({n_toks=})")

    try:
        out = await call(this_prompt, t=t, n=n, max_n_per_call=max_n_per_round)
    except Exception as e:
        print(f"{name=}")

        if isinstance(e, RuntimeError) and "unsafe content" in str(e):
            return name, all_reasoning_and_outputs, args, None
        if isinstance(e, RuntimeError) and name == "7c9b52a0.json":
            print("hack for now!")
            return name, all_reasoning_and_outputs, args, None
        if isinstance(e, ContextLengthExceeded):
            return name, all_reasoning_and_outputs, args, None

        raise
    print(f"{out[0].token_usage=}")

    return (
        name,
        all_reasoning_and_outputs,
        args,
        [x.completion for x in out],
    )


# Idea with this was to special case examples where grids are equal size, but substantially different colors (not is_smallish_edit(...))
# However, there are only ~5% of examples like this and they probably aren't handled that poorly by our existing functions IMO.
def is_smallish_edit(name: str, max_avg_diff: float = 0.4, max_any_diff: float = 0.6):
    train_data = out_data_by_name_d[name]["train"]

    all_average_differing = []
    for d in train_data:
        input_grid = np.array(d["input"])
        output_grid = np.array(d["output"])

        if input_grid.shape != output_grid.shape:
            return False

        this_average_differing = np.mean(input_grid != output_grid)

        if this_average_differing > max_any_diff:
            return False

        all_average_differing.append(this_average_differing)

    return np.mean(all_average_differing) < max_avg_diff
