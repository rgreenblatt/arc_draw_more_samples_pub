import os
import json

# Load the JSON content
json_file_path = os.environ.get(
    "INPUT_JSON", "/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json"
)
with open(json_file_path, "r") as file:
    out_data_by_name_d = json.load(file)

loaded_names = list(out_data_by_name_d.keys())

train_json_file_path = os.environ.get(
    "INPUT_TRAIN_JSON", "/kaggle/input/arc-prize-2024/arc-agi_training_challenges.json"
)

with open(train_json_file_path, "r") as file:
    train_data = json.load(file)

# backward compatibility hack
out_data_by_name_d.update({name + ".json": v for name, v in train_data.items()})
