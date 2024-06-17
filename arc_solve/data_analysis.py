import numpy as np
import matplotlib.pyplot as plt

from arc_solve.load_data import out_eval_data_by_name, out_train_data_by_name

# %%

eval_sizes = [np.array(x["input"]).size for item in out_eval_data_by_name for x in item[1]["train"]]
train_sizes = [np.array(x["input"]).size for item in out_train_data_by_name for x in item[1]["train"]]

# %%

plt.hist(eval_sizes, bins=10, alpha=0.5, label="eval", density=True)
plt.hist(train_sizes, bins=10, alpha=0.5, label="train", density=True)

