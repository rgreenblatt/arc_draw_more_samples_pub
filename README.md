# ARC Draw more samples

See [my blog post on substack for details](https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt).

The main script is arc_solve/solve.py. This requires a redis server running at port "6381" (for caching), several hours of time, a considerable amount of memory, an openai key, and a bunch of money.

Required dependencies are tqdm, numpy, scipy, skimage, attrs, cattrs, nest_asyncio, redis-py, matplotlib, anthropic (not actually needed), and openai==0.28.1. (I might make an actual requirements.txt file later.)

Data can be loaded from jsons which can be found [at this drive link](https://drive.google.com/file/d/1t3LmW0oxnRHTksgeUrMwPYZMZ8dOb4X4/view?usp=sharing) and visualized and viewed using arc_solve/load_and_viz.py. (I may add additional plots and display later.)
