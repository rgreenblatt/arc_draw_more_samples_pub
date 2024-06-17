import asyncio
import logging
import os
import random
import threading
import time
import uuid
from collections import defaultdict
from itertools import cycle
from typing import Optional, Union

import attrs
import openai
import requests
import tiktoken
from openai.openai_object import OpenAIObject as OpenAICompletion
from termcolor import cprint

from rrutils.llm_api.base_llm import (
    PRINT_COLORS,
    ContextLengthExceeded,
    LLMResponse,
    ModelAPIProtocol,
    StopReason,
    TokenUsage,
)

OAIChatPrompt = list[dict[str, str]]
OAIBasePrompt = Union[str, list[str]]

DISABLE_POST = True


def post_json_in_background(url, json_data):
    if DISABLE_POST:
        return

    def send_request():
        try:
            response = requests.post(url, json=json_data, timeout=0.1)
            # print(f"Response: {response.json()}")
        except requests.RequestException:
            pass  # Do nothing on failure

    # Create and start a thread to run the send_request function
    thread = threading.Thread(target=send_request)
    thread.start()


def count_tokens(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def count_chat_tokens(prompt: OAIChatPrompt) -> int:
    """
    According to https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    - every message follows <|start|>{role}\n{content}<|end|>\n (4 extra tokens)
    - every reply is primed with <|start|>assistant<|message|> (3 extra tokens)

    This may be out of date for models beyond 0613.
    """
    if any(isinstance(x["content"], list) for x in prompt):
        return -1000
    return sum(count_tokens(x["content"]) + 4 for x in prompt) + 3


def price_per_token(model_id: str) -> tuple[float, float]:
    """
    Returns the (input token, output token) price for the given model id.
    """
    if model_id.startswith("gpt-4-32k"):
        prices = 0.06, 0.12
    elif model_id.startswith("gpt-4-turbo"):
        prices = 0.01, 0.03
    elif model_id.startswith("gpt-4o"):
        prices = 0.005, 0.015
    elif model_id.startswith("gpt-4-1106"):
        prices = 0.01, 0.03
    elif model_id.startswith("gpt-4-0125"):
        prices = 0.01, 0.03
    elif model_id.startswith("ft:gpt-4-0613"):
        prices = 0.06, 0.12  # idk if this is right
    elif model_id.startswith("gpt-4"):
        prices = 0.03, 0.06
    elif model_id.startswith("gpt-3.5-turbo-16k"):
        prices = 0.003, 0.004
    elif model_id.startswith("gpt-3.5-turbo"):
        prices = 0.0015, 0.002
    elif model_id.startswith("ft:gpt-3.5-turbo"):
        prices = 0.003, 0.006  # idk if this is right
    elif model_id == "davinci-002":
        prices = 0.0004, 0.0004
    elif model_id == "babbage-002":
        prices = 0.0002, 0.0002
    elif model_id == "text-davinci-003":
        prices = 0.02, 0.02
    else:
        raise ValueError(f"Invalid model id: {model_id}")

    return tuple(price / 1000 for price in prices)


@attrs.define()
class Resource:
    """
    A resource that is consumed over time and replenished at a constant rate.
    """

    refresh_rate: float = attrs.field()  # How many units of the resource are replenished per minute
    value: float = attrs.field(init=False)
    last_update_time: float = attrs.field(init=False, factory=time.time)

    def __attrs_post_init__(self):
        self.value = self.refresh_rate

    def _replenish(self):
        """
        Updates the value of the resource based on the time since the last update.
        """
        curr_time = time.time()
        self.value = min(self.refresh_rate, self.value + (curr_time - self.last_update_time) * self.refresh_rate / 60)
        self.last_update_time = curr_time

    def geq(self, amount: float) -> bool:
        self._replenish()
        return self.value >= amount

    def consume(self, amount: float):
        """
        Consumes the given amount of the resource.
        """
        assert self.geq(amount), f"Resource does not have enough capacity to consume {amount} units"
        self.value -= amount


# TODO: handle the following error correctly
# "WARNING:root:Encountered API error: Invalid key in 'logit_bias':
# . You should only be submitting non-negative integers..
# Retrying now. (Attempt 1)"
@attrs.define
class OpenAIModel(ModelAPIProtocol):
    frac_rate_limit: float
    model_ids: set[str] = attrs.field(init=False, default=attrs.Factory(set))

    # rate limit
    token_capacity: dict[str, Resource] = attrs.field(init=False, default=attrs.Factory(dict))
    request_capacity: dict[str, Resource] = attrs.field(init=False, default=attrs.Factory(dict))

    @staticmethod
    def _assert_valid_id(model_id: str):
        raise NotImplementedError

    @staticmethod
    async def _get_dummy_response_header(model_id: str):
        raise NotImplementedError

    @staticmethod
    def _count_prompt_token_capacity(prompt, **kwargs) -> int:
        raise NotImplementedError

    async def _make_api_call(self, prompt, model_id, **params) -> list[LLMResponse]:
        raise NotImplementedError

    @staticmethod
    def _print_prompt_and_response(prompt, responses):
        raise NotImplementedError

    async def add_model_id(self, model_id: str, attempts: int = 50):
        self._assert_valid_id(model_id)
        if model_id in self.model_ids:
            return

        # make dummy request to get token and request capacity
        for i in range(attempts):
            try:
                if model_id in self.model_ids:
                    return
                model_metadata = await self._get_dummy_response_header(model_id)
                token_capacity = int(model_metadata["x-ratelimit-limit-tokens"])
                request_capacity = int(model_metadata["x-ratelimit-limit-requests"])
                tokens_consumed = token_capacity - int(model_metadata["x-ratelimit-remaining-tokens"])
                request_consumed = request_capacity - int(model_metadata["x-ratelimit-remaining-requests"])
                token_capacity = Resource(token_capacity * self.frac_rate_limit)
                request_capacity = Resource(request_capacity * self.frac_rate_limit)
                token_capacity.consume(tokens_consumed)
                request_capacity.consume(request_consumed)
                self.token_capacity[model_id] = token_capacity
                self.request_capacity[model_id] = request_capacity
                self.model_ids.add(model_id)
                return
            except Exception as e:
                logging.warn(f"Failed to get dummy response header for {model_id} (attempt {i}): {e}")
                await asyncio.sleep((i + 2) * random.random())
        raise RuntimeError(f"Failed to get dummy response header for {model_id} after {attempts} attempts.")

    async def __call__(
        self,
        model_ids: list[str],
        prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        stream_per_chunk_timeout: Optional[float] = 20.0,  # could probably be made much more aggressive if we wanted # NOTE: Default is actually in llm.py
        non_streaming_timeout: Optional[float] = 600.0,
        **kwargs,
    ) -> list[LLMResponse]:
        # get random uuid
        assert len(model_ids) == 1, "hack"
        call_id = str(uuid.uuid1())

        async def attempt_api_call():
            for model_id in cycle(model_ids):
                request_capacity, token_capacity = self.request_capacity[model_id], self.token_capacity[model_id]
                if request_capacity.geq(1) and token_capacity.geq(token_count):
                    request_capacity.consume(1)
                    token_capacity.consume(token_count)
                    print(f"{model_id=} {token_capacity.value=} {request_capacity.value=}")
                    return await self._make_api_call(
                        prompt, model_id, stream_per_chunk_timeout=stream_per_chunk_timeout, non_streaming_timeout=non_streaming_timeout, **kwargs
                    )
                else:
                    await asyncio.sleep(0.01)

        model_ids.sort(key=lambda model_id: price_per_token(model_id)[0])  # Default to cheapest model
        await asyncio.gather(*[self.add_model_id(model_id) for model_id in model_ids])
        token_count = self._count_prompt_token_capacity(prompt, **kwargs)
        # print(f"{self.token_capacity=}")
        assert (
            max(self.token_capacity[model_id].refresh_rate for model_id in model_ids) >= token_count
        ), f"Prompt is too long for any model to handle.\n\n{prompt}"
        responses: Optional[list[LLMResponse]] = None

        post_json_in_background(
            "http://0.0.0.0:8944/report_start_request",
            json_data={"id": call_id, "model_ids": model_ids, "prompt": prompt},
        )

        _max_attempts = max_attempts
        i = 0
        while i < _max_attempts:
            try:
                responses = await attempt_api_call()
                break
            except Exception as e:
                if "input image may contain content that is not allowed by our safety system" in str(e):
                    logging.error(f"Detected unsafe content in prompt, immediately erroring!" + str(e))
                    raise RuntimeError("Detected unsafe content in prompt")

                if "This model's maximum context length is " in str(e):
                    # print("DEBUG_PROMPT\n", prompt)
                    logging.error(f"Context length exceeded, immediately erroring!" + str(e))
                    raise ContextLengthExceeded(e)
                if "repetitive patterns in your prompt" in str(e):
                    return [
                        LLMResponse(model_id=model_id, completion="", stop_reason=StopReason.WEIRD, cost=0)
                        for model_id in model_ids
                    ]

                if "Rate limit reached" in str(e):
                    _max_attempts += 1

                logging.warn(f"Encountered API error: {str(e)}.\nRetrying now. (Attempt {i})")


                if isinstance(e, openai.error.OpenAIError):
                    if isinstance(e, openai.error.RateLimitError):
                        self.token_capacity[model_ids[0]]._replenish()
                        self.token_capacity[model_ids[0]].value = -self.token_capacity[model_ids[0]].refresh_rate * (1- self.frac_rate_limit)
                        print(f"RATE LIMIT REACHED, RESETTING TOKEN CAPACITY {model_ids[0]}")
                    else:
                        try:

                            if hasattr(e, "request") and "x-ratelimit-remaining-tokens" in e.request.headers:
                                self.token_capacity[model_ids[0]]._replenish()
                                self.token_capacity[model_ids[0]].value = int(e.request.headers["x-ratelimit-remaining-tokens"]) - self.token_capacity[model_ids[0]].refresh_rate * (1- self.frac_rate_limit)
                                self.request_capacity[model_ids[0]].value = int(e.request.headers["x-ratelimit-remaining-requests"]) - self.request_capacity[model_ids[0]].refresh_rate * (1- self.frac_rate_limit)
                        except Exception as e:
                            print("FAILED TO UPDATE CAPACITY", e)

                await asyncio.sleep(min(1.5**i * random.random(), 60))

                i += 1

        if responses is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        if print_prompt_and_response:
            self._print_prompt_and_response(prompt, responses)
        post_json_in_background(
            "http://0.0.0.0:8944/report_request_success",
            json_data={"id": call_id, "responses": [r.as_dict() for r in responses]},
        )
        return responses


_GPT_4_MODELS = [
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "gpt-4o",
]
_GPT_TURBO_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-0125",
]
GPT_CHAT_MODELS = set(_GPT_4_MODELS + _GPT_TURBO_MODELS)


def choice_logprobs_to_l_dict(logprobs) -> list[dict[str, float]]:
    if logprobs is None:
        return []
    return [{d["token"]: d["logprob"] for d in l.top_logprobs} for l in logprobs.content]


@attrs.define
class RunningCompletion:
    content_deltas: list[str] = attrs.Factory(list)
    logprobs: list[dict[str, float]] = attrs.Factory(list)
    finish_reason: Optional[str] = None

    def accumulate(self):
        assert self.finish_reason is not None, "not actually finished!"
        return AccumulatedCompletion(
            "".join(self.content_deltas),
            StopReason.factory(self.finish_reason),
            logprobs=self.logprobs if self.logprobs else None,
        )


@attrs.define
class AccumulatedCompletion:
    completion: str
    finish_reason: StopReason
    logprobs: Optional[list[dict[str, float]]] = None

    @classmethod
    def from_choice(cls, choice: OpenAICompletion):
        return cls(
            completion=choice.message.content,
            finish_reason=StopReason.factory(choice.finish_reason),
            logprobs=choice_logprobs_to_l_dict(choice.logprobs),
        )


class OpenAIChatModel(OpenAIModel):
    def _assert_valid_id(self, model_id: str):
        if model_id.startswith("ft:"):
            _, model_name, *_ = model_id.split(":")
            assert model_name in GPT_CHAT_MODELS, f"Invalid ft model id: {model_id}"
            return
        assert model_id in GPT_CHAT_MODELS, f"Invalid model id: {model_id}"

    async def _get_dummy_response_header(self, model_id: str):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        if model_id.startswith("ft:gpt-3.5-turbo"):
            model_id = model_id.split(":")[1]
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Say 1"}],
            "max_tokens": 1,
        }
        response = requests.post(url, headers=headers, json=data)
        return response.headers

    @staticmethod
    def _count_prompt_token_capacity(prompt: OAIChatPrompt, **kwargs) -> int:
        # The magic formula is: .25 * (total number of characters) + (number of messages) + (max_tokens, or 15 if not specified)
        BUFFER = 5  # A bit of buffer for some error margin
        MIN_NUM_TOKENS = 20

        num_tokens = 0
        for message in prompt:
            num_tokens += 1
            num_tokens += len(message["content"]) / 4

        max_tokens = kwargs.get("max_tokens", 15)
        if max_tokens is None:
            max_tokens = 15
        return max(MIN_NUM_TOKENS, int(num_tokens + BUFFER) + kwargs.get("n", 1) * min(max_tokens, 2000)) # obviously very hacky

    async def _make_api_call(
        self, prompt: OAIChatPrompt, model_id, stream_per_chunk_timeout: Optional[float] = None, non_streaming_timeout:Optional[float] = 600.0, **params
    ) -> list[LLMResponse]:
        # stream so we can do a more precise timeout
        coroutine = openai.ChatCompletion.acreate(
            messages=prompt, model=model_id, stream=stream_per_chunk_timeout is not None, **params
        )
        if stream_per_chunk_timeout is not None:
            api_response: OpenAICompletion = await asyncio.wait_for(coroutine, timeout=stream_per_chunk_timeout)  # type: ignore
            completions = await self._handle_streaming_api_call(
                api_response, stream_per_chunk_timeout=stream_per_chunk_timeout
            )

            prompt_tokens = count_chat_tokens(prompt)
        else:
            api_response: OpenAICompletion = await asyncio.wait_for(coroutine, timeout=non_streaming_timeout)  # type: ignore
            completions = [AccumulatedCompletion.from_choice(c) for c in api_response.choices]

            prompt_tokens = api_response.usage.prompt_tokens

        context_token_cost, completion_token_cost = price_per_token(model_id)
        context_cost = prompt_tokens * context_token_cost

        return [
            LLMResponse(
                model_id=model_id,
                completion=c.completion,
                stop_reason=c.finish_reason,
                cost=context_cost + count_tokens(c.completion) * completion_token_cost, # TODO: support new gpt-4o tokenizer
                logprobs=c.logprobs,
                token_usage=TokenUsage(input_tokens=prompt_tokens, output_tokens=count_tokens(c.completion)),
            )
            for c in completions
        ]

    async def _handle_streaming_api_call(
        self, stream_out: OpenAICompletion, stream_per_chunk_timeout: Optional[float] = None
    ):
        completion_item_by_index: defaultdict[str, RunningCompletion] = defaultdict(RunningCompletion)
        while True:
            try:
                n = anext(stream_out)  # type: ignore
                item: OpenAICompletion = await asyncio.wait_for(n, timeout=stream_per_chunk_timeout)
                assert len(item.choices) == 1
                completion = completion_item_by_index[item.choices[0].index]
                completion.content_deltas.append(item.choices[0].delta.get("content", ""))
                completion.finish_reason = item.choices[0].finish_reason
                if item.choices[0].logprobs is not None:
                    completion.logprobs.extend(choice_logprobs_to_l_dict(item.choices[0].logprobs))
            except StopAsyncIteration:
                break

        return [completion_item_by_index[idx].accumulate() for idx in sorted(completion_item_by_index.keys())]

    @staticmethod
    def _print_prompt_and_response(prompts: OAIChatPrompt, responses: list[LLMResponse]):
        for prompt in prompts:
            role, text = prompt["role"], prompt["content"]
            cprint(f"=={role.upper()}:", "white")
            cprint(text, PRINT_COLORS[role])
        for i, response in enumerate(responses):
            if len(responses) > 1:
                cprint(f"==RESPONSE {i + 1} ({response.model_id}):", "white")
            cprint(response.completion, PRINT_COLORS["assistant"], attrs=["bold"])
        print()


BASE_MODELS = {"davinci-002", "babbage-002", "text-davinci-003", "gpt-3.5-turbo-instruct"}


class OpenAIBaseModel(OpenAIModel):
    def _assert_valid_id(self, model_id: str):
        assert model_id in BASE_MODELS, f"Invalid model id: {model_id}"

    async def _get_dummy_response_header(self, model_id: str):
        url = "https://api.openai.com/v1/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        data = {"model": model_id, "prompt": "a", "max_tokens": 1}
        response = requests.post(url, headers=headers, json=data)
        return response.headers

    @staticmethod
    def _count_prompt_token_capacity(prompt: OAIBasePrompt, **kwargs) -> int:
        max_tokens = kwargs.get("max_tokens", 15)
        n = kwargs.get("n", 1)
        completion_tokens = n * max_tokens

        tokenizer = tiktoken.get_encoding("cl100k_base")
        if isinstance(prompt, str):
            prompt_tokens = len(tokenizer.encode(prompt))
            return prompt_tokens + completion_tokens
        else:
            prompt_tokens = sum(len(tokenizer.encode(p)) for p in prompt)
            return prompt_tokens + completion_tokens

    async def _make_api_call(
        self, prompt: OAIBasePrompt, model_id, stream_per_chunk_timeout: Optional[float] = None, **params
    ) -> list[LLMResponse]:
        if stream_per_chunk_timeout is not None:
            ...  # do nothing for now, should be implemented later
            # raise NotImplementedError()
        api_response: OpenAICompletion = await openai.Completion.acreate(prompt=prompt, model=model_id, **params)  # type: ignore
        context_token_cost, completion_token_cost = price_per_token(model_id)
        context_cost = api_response.usage.prompt_tokens * context_token_cost
        return [
            LLMResponse(
                model_id=model_id,
                completion=choice.text,
                stop_reason=choice.finish_reason,
                cost=context_cost + count_tokens(choice.text) * completion_token_cost,
                logprobs=choice.logprobs.top_logprobs if choice.logprobs is not None else None,
            )
            for choice in api_response.choices
        ]

    @staticmethod
    def _print_prompt_and_response(prompt: OAIBasePrompt, responses: list[LLMResponse]):
        prompt_list = prompt if isinstance(prompt, list) else [prompt]
        responses_per_prompt = len(responses) // len(prompt_list)
        responses_list = [
            responses[i : i + responses_per_prompt] for i in range(0, len(responses), responses_per_prompt)
        ]
        for i, (prompt, response_list) in enumerate(zip(prompt_list, responses_list)):
            if len(prompt_list) > 1:
                cprint(f"==PROMPT {i + 1}", "white")
            if len(response_list) == 1:
                cprint(f"=={response_list[0].model_id}", "white")
                cprint(prompt, PRINT_COLORS["user"], end="")
                cprint(response_list[0].completion, PRINT_COLORS["assistant"], attrs=["bold"])
            else:
                cprint(prompt, PRINT_COLORS["user"])
                for j, response in enumerate(response_list):
                    cprint(f"==RESPONSE {j + 1} ({response.model_id}):", "white")
                    cprint(response.completion, PRINT_COLORS["assistant"], attrs=["bold"])
            print()
