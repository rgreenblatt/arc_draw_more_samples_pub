import numpy as np
import json

from tqdm.asyncio import tqdm_asyncio

from arc_solve.prompting import call, gpt4_o_image_test_few_shot, print_prompt
from arc_solve.render import RenderArgs

# %%


async def get_random_3_shot(
    idx: int,
    shape=(5, 5),
    render_args: RenderArgs = RenderArgs(use_alt_color_scheme=True),
    print_p: bool = False,
):
    generator = np.random.RandomState(idx)
    random_image = generator.randint(0, 10, shape)
    examples = [generator.randint(0, 10, shape) for _ in range(5)]

    prompt = gpt4_o_image_test_few_shot(random_image, examples, render_args=render_args)

    if print_p:
        print_prompt(prompt, show_images=True)

    out = await call(prompt, t=0.0, n=1, max_tokens=1000)

    color_to_num = {
        "black": 0,
        "blue": 1,
        "red": 2,
        "green": 3,
        "yellow": 4,
        "grey": 5,
        "pink": 6,
        "orange": 7,
        "purple": 8,
        "brown": 9,
    }

    try:
        actual = np.array(
            [[color_to_num[x] for x in y] for y in json.loads(out[0].completion)]
        )
    except ValueError:
        return 0.0, np.zeros(10), np.zeros(10)

    if actual.shape != shape:
        return 0.0, np.zeros(10), np.zeros(10)

    eq_vals = random_image == actual

    count_corr_by_color = np.zeros(10)
    count_color = np.zeros(10)

    for color in color_to_num.values():
        count_corr_by_color[color] = eq_vals[random_image == color].sum()
        count_color[color] = (random_image == color).sum()

    return eq_vals.mean(), count_corr_by_color, count_color


# %%


# await get_random_3_shot(
#     0,
#     render_args=RenderArgs(
#         cell_size=105,
#         use_alt_color_scheme=True,
#         use_border=False,
#         use_larger_edges=True,
#         force_edge_size=19,
#     ),
#     print_p=True,
#     shape=(8, 8),
# )

# %%


many_outs = await tqdm_asyncio.gather(
    *[
        get_random_3_shot(
            idx,
            render_args=RenderArgs(
                cell_size=40,
                use_alt_color_scheme=True,
                # use_border=False,
                # use_larger_edges=True,
                # force_edge_size=10,
                # force_high_res=True,
            ),
            shape=(11, 10),
        )
        for idx in range(50)
    ]
)

# %%

np.mean([acc for acc, _, _ in many_outs])

# %%

total_corr_by_color = np.zeros(10)
total_color = np.zeros(10)

for _, corr_by_color, color in many_outs:
    total_corr_by_color += corr_by_color
    total_color += color

acc_by_color = total_corr_by_color / total_color

np.argmin(acc_by_color)
