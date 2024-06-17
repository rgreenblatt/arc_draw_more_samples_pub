import os
import datetime
import time
import requests

# %%

import openai

# setup key to be one with fine-tuning access
if "OPENAI_API_KEY" not in os.environ:
    openai.api_key_path = os.path.expanduser("~/.openai_api_key")  # NOTE: _alt!!!
    try:
        with open(openai.api_key_path, "r") as f:
            os.environ["OPENAI_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        ...

# %%

# %%


def get_remaining_tpm(model_id: str):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    data = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Say 1"}],
        "max_tokens": 1,
    }
    response = requests.post(url, headers=headers, json=data)
    headers = response.headers

    return headers["x-ratelimit-remaining-tokens"]


# %%

pretty_format_datetime = lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
while True:
    model_id = "gpt-4o"
    remaining_tpm = get_remaining_tpm(model_id)
    print(f"{remaining_tpm=} {pretty_format_datetime()=}")
    time.sleep(10)
