import os
import json

# Load the JSON content
json_file_path = os.environ.get(
    "INPUT_JSON", "/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json"
)
with open(json_file_path, "r") as file:
    out_data_by_name_d = json.load(file)

loaded_names = list(out_data_by_name_d.keys())

for file in os.listdir("ARC-AGI/data/training"):
    with open("ARC-AGI/data/training/" + file) as f:
        data = json.load(f)
        out_data_by_name_d[file] = data
