import asyncio
import math
import os
import datetime
import json
import logging
import random
from collections import defaultdict
from itertools import chain
from typing import Callable, Optional, Union

import attrs

from rrutils.llm_api.anthropic_llm import (
    ANTHROPIC_MODELS,
    ANTHROPIC_NEW_MODELS,
    AnthropicChatModel,
    AnthropicNewChatModel,
    AnthropicNewChatPrompt,
    AnthropicMessageParam,
)
from rrutils.llm_api.base_llm import LLMResponse, ModelAPIProtocol, StopReason
from rrutils.llm_api.openai_llm import (
    BASE_MODELS,
    GPT_CHAT_MODELS,
    OAIBasePrompt,
    OAIChatPrompt,
    OpenAIBaseModel,
    OpenAIChatModel,
)
from rrutils.redis_cache_wrapper import RedisWrapper, cache_key, get_digest


def sanitize_dict_for_hash(d: dict | AnthropicMessageParam):
    assert isinstance(d, dict)
    return sorted(d.items(), key=lambda x: x[0])


def sanitize_oaichat_or_anthropic_prompt(
    prompt: OAIChatPrompt | AnthropicNewChatPrompt,
):
    return [sanitize_dict_for_hash(x) for x in prompt]


@attrs.define()
class ModelAPI:
    anthropic_num_threads: int = 5  # current redwood limit is 5
    openai_fraction_rate_limit: float = (
        0.99  # pick a number < 1 to avoid hitting the rate limit
    )

    _openai_base: OpenAIBaseModel = attrs.field(init=False)
    _openai_chat: OpenAIChatModel = attrs.field(init=False)
    _anthropic_chat: AnthropicChatModel = attrs.field(init=False)
    _anthropic_new_chat: AnthropicNewChatModel = attrs.field(init=False)

    def __attrs_post_init__(self):
        self._openai_base = OpenAIBaseModel(
            frac_rate_limit=self.openai_fraction_rate_limit
        )
        self._openai_chat = OpenAIChatModel(
            frac_rate_limit=self.openai_fraction_rate_limit
        )
        self._anthropic_chat = AnthropicChatModel(
            num_threads=self.anthropic_num_threads
        )
        self._anthropic_new_chat = AnthropicNewChatModel(
            num_threads=self.anthropic_num_threads
        )

    async def call_cached(
        self,
        model_ids: Union[str, list[str]],
        prompt,
        max_tokens: Optional[int],
        print_prompt_and_response=False,
        t: float = 0.0,
        n: int = 1,
        max_n_per_call: Optional[int] = 128,
        max_attempts_per_api_call=10,
        deterministic: bool = True,
        compute_fresh: bool = False,
        clear_and_compute_fresh: bool = False,
        stream_per_chunk_timeout: Optional[
            float
        ] = 20.0,  # could probably be made much more aggressive if we wanted
        non_streaming_timeout: Optional[float] = 240.0,
        **kwargs,
    ) -> list[LLMResponse]:
        if isinstance(model_ids, str):
            model_ids = [model_ids]

        cached_results: list[LLMResponse] = []
        required_samples = n

        if isinstance(prompt, list) and isinstance(prompt[0], dict):
            prompt_sanitized = sanitize_oaichat_or_anthropic_prompt(prompt)
        elif isinstance(prompt, str):
            prompt_sanitized = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")
        function_input_sanitized = [
            prompt_sanitized,
            sanitize_dict_for_hash(kwargs),
            t,
            max_tokens,
        ]
        base_key = cache_key(function_input_sanitized, "model_api_call")

        def get_cached_results_key(supplier: str):
            supplier_key = base_key + f":{supplier}"
            return f"cached_results:{supplier_key}"

        # Check cache for each supplier and collect cached results
        for supplier in sorted(model_ids):
            cached_results_key = get_cached_results_key(supplier)

            if clear_and_compute_fresh:
                await RedisWrapper.singleton().clear(cached_results_key)

            if not compute_fresh:
                supplier_cached_results = await RedisWrapper.singleton().lrange(
                    cached_results_key, 0, required_samples - 1
                )
                supplier_cached_results = [
                    LLMResponse.from_dict(json.loads(result))
                    for result in supplier_cached_results
                ]
                cached_results.extend(supplier_cached_results)
                required_samples -= len(supplier_cached_results)

        ban_uncached_queries = os.getenv("BAN_UNCACHED_LLM_QUERIES", "0") == "1"

        # If not enough cached results, call the original function
        if required_samples > 0:
            if ban_uncached_queries:
                raise ValueError("Uncached queries are banned.")

            if max_n_per_call is None:
                counts_per_call = [required_samples]
            else:
                counts_per_call = []
                total_so_far = 0

                assert max_n_per_call > 0

                for _ in range(math.ceil(required_samples / max_n_per_call)):
                    this_call_count = min(
                        max_n_per_call, required_samples - total_so_far
                    )
                    assert this_call_count > 0
                    assert this_call_count <= max_n_per_call

                    counts_per_call.append(this_call_count)
                    total_so_far += this_call_count

                assert total_so_far == required_samples

            async def call_and_write_to_cache(
                this_call_count: int,
            ):
                new_results_here = await self(
                    model_ids,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=t,
                    n=this_call_count,
                    print_prompt_and_response=print_prompt_and_response,
                    max_attempts_per_api_call=max_attempts_per_api_call,
                    stream_per_chunk_timeout=stream_per_chunk_timeout,
                    non_streaming_timeout=non_streaming_timeout,
                    **kwargs,
                )

                if not isinstance(new_results_here, list):
                    new_results_here = [new_results_here]

                # function_input_sanitized_for_no_limit = [prompt_sanitized, sanitize_dict_for_hash(kwargs), t, None]
                # base_key_no_limit = cache_key(function_input_sanitized_for_no_limit, "model_api_call")

                # def get_cached_results_key_no_limit(supplier: str):
                #     supplier_key = base_key_no_limit + f":{supplier}"
                #     return f"cached_results:{supplier_key}"

                # Cache the new results under their respective suppliers
                if len(new_results_here) > 0:
                    print(f"cache write {new_results_here[0].token_usage=}")
                for result in new_results_here:
                    supplier = result.model_id
                    cached_results_key = get_cached_results_key(supplier)
                    result_as_str = json.dumps(result.as_dict())
                    await RedisWrapper.singleton().rpush(
                        cached_results_key, result_as_str
                    )

                    # this is vulnerable to selection bias, so probably we don't want this...
                    # if result.stop_reason != StopReason.MAX_TOKENS and max_tokens is not None: # also push to None if we didn't get limited???
                    #     cached_results_key_no_limit = get_cached_results_key_no_limit(supplier)
                    #     await RedisWrapper.singleton().rpush(cached_results_key_no_limit, result_as_str)

                return new_results_here

            # maybe multi dispatch should happen one layer lower? # TODO
            new_results_multi = await asyncio.gather(
                *(
                    call_and_write_to_cache(this_call_count)
                    for this_call_count in counts_per_call
                )
            )
            new_results_multi_as_lst = [
                x if isinstance(x, list) else [x] for x in new_results_multi
            ]
            new_results = list(chain.from_iterable(new_results_multi_as_lst))

            cached_results.extend(new_results)

        if deterministic or len(cached_results) < n:
            return cached_results[:n]
        else:
            return random.sample(cached_results, n)

    async def __call__(
        self,
        model_ids: Union[str, list[str]],
        prompt: Union[OAIBasePrompt, OAIChatPrompt, str],
        max_tokens: Optional[int],
        print_prompt_and_response: bool = False,
        n: int = 1,
        max_attempts_per_api_call: int = 10,
        max_attempts_for_valid_completion: int = 1,
        is_valid: Callable[[str], bool] = lambda _: True,
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Make maximally efficient API requests for the specified model(s) and prompt.

        Args:
            model_ids: The model(s) to call. If multiple models are specified, the output will be sampled from the
                cheapest model that has capacity. All models must be from the same class (e.g. OpenAI Base,
                OpenAI Chat, or Anthropic Chat). Anthropic chat will error if multiple models are passed in.
                Passing in multiple models could speed up the response time if one of the models is overloaded.
            prompt: The prompt to send to the model(s). Type should match what's expected by the model(s).
            max_tokens: The maximum number of tokens to request from the API (argument added to
                standardize the Anthropic and OpenAI APIs, which have different names for this).
            print_prompt_and_response: Whether to print the prompt and response to stdout.
            n: The number of completions to request.
            max_attempts_per_api_call: Passed to the underlying API call. If the API call fails (e.g. because the
                API is overloaded), it will be retried this many times. If still fails, an exception will be raised.
            max_attempts_for_valid_completion: If a completion is rejected by the is_valid predicate, it will be
                resampled this many times. If no valid completions are found, an exception will be raised. Up to
                n * max_attempts_for_valid_completion completions may be requested from the API.
            is_valid: Completions are rejection sampled until one is found that satisfies this predicate.
        """
        assert (
            "max_tokens_to_sample" not in kwargs
        ), "max_tokens_to_sample should be passed in as max_tokens."

        if isinstance(model_ids, str):
            model_ids = [model_ids]

        def model_id_to_class(model_id: str) -> ModelAPIProtocol:
            if model_id in BASE_MODELS:
                return self._openai_base
            elif model_id in GPT_CHAT_MODELS:
                return self._openai_chat
            elif model_id in ANTHROPIC_NEW_MODELS:
                return self._anthropic_new_chat
            elif model_id in ANTHROPIC_MODELS:
                return self._anthropic_chat
            elif (
                model_id.startswith("ft:") and model_id.split(":")[1] in GPT_CHAT_MODELS
            ):
                return self._openai_chat
            raise ValueError(f"Invalid model id: {model_id}")

        model_classes = [model_id_to_class(model_id) for model_id in model_ids]
        if len(set(str(type(x)) for x in model_classes)) != 1:
            raise ValueError("All model ids must be of the same type.")

        model_class = model_classes[0]
        if isinstance(model_class, AnthropicChatModel):
            kwargs["max_tokens_to_sample"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        num_remaining_attempts = max_attempts_for_valid_completion * n
        valid_responses = []
        # TODO: handling n in this way is inefficient!!!
        while (
            num_needed := n - len(valid_responses)
        ) > 0 and num_remaining_attempts > 0:
            num_remaining_attempts -= num_needed
            if isinstance(model_class, AnthropicChatModel):
                assert isinstance(
                    prompt, str
                ), "Anthropic models (old) only support string prompts."
                responses = list(
                    chain.from_iterable(
                        await asyncio.gather(
                            *[
                                model_class(
                                    model_ids,
                                    prompt,
                                    print_prompt_and_response,
                                    max_attempts_per_api_call,
                                    **kwargs,
                                )
                                for _ in range(num_needed)
                            ]
                        )
                    )
                )
            elif isinstance(model_class, AnthropicNewChatModel):
                assert not isinstance(
                    prompt, str
                ), "Anthropic models (new) only support list prompts."
                responses = list(
                    chain.from_iterable(
                        await asyncio.gather(
                            *[
                                model_class(
                                    model_ids,
                                    prompt,
                                    print_prompt_and_response=print_prompt_and_response,
                                    max_attempts=max_attempts_per_api_call,
                                    **kwargs,
                                )
                                for _ in range(num_needed)
                            ]
                        )
                    )
                )
            else:
                responses = await model_class(
                    model_ids,
                    prompt,
                    print_prompt_and_response,
                    max_attempts_per_api_call,
                    n=num_needed,
                    **kwargs,
                )
            valid_responses.extend(
                [response for response in responses if is_valid(response.completion)]
            )

        num_sampled = max_attempts_for_valid_completion * n - num_remaining_attempts
        num_valid = len(valid_responses)
        logging.info(
            f"`is_valid` success rate: {(1 - (num_valid / num_sampled)) * 100:.2f}%"
        )
        if num_valid < n:
            raise RuntimeError(
                f"Only found {num_valid}/{n} valid completions after {num_sampled} attempts."
            )
        return valid_responses


async def demo():
    model_api = ModelAPI(anthropic_num_threads=2, openai_fraction_rate_limit=1)
    # anthropic_requests = [
    #     model_api("claude-instant-1", "\n\nHuman: What's your name?\n\nAssistant:", max_tokens=2)
    # ]
    oai_chat_messages = [
        [
            {"role": "system", "content": "You are a comedic pirate."},
            {"role": "user", "content": "Hello!"},
        ],
        [
            {
                "role": "system",
                "content": "You are a swashbuckling space-faring voyager.",
            },
            {"role": "user", "content": "Hello!"},
        ],
    ]
    oai_chat_models = ["gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613"]
    oai_chat_requests = [
        model_api(
            oai_chat_models,
            prompt=message,
            n=6,
            max_tokens=200,
            print_prompt_and_response=True,
        )
        for message in oai_chat_messages
    ]
    oai_chat_requests_cached = [
        model_api.call_cached(
            oai_chat_models,
            prompt=message,
            n=6,
            max_tokens=200,
            print_prompt_and_response=True,
        )
        for message in oai_chat_messages
    ]
    oai_messages = ["1 2 3", ["beforeth they cometh", "whence afterever the storm"]]
    oai_models = ["davinci-002"]
    oai_requests = [
        model_api(
            oai_models,
            prompt=message,
            max_tokens=200,
            n=1,
            print_prompt_and_response=True,
        )
        for message in oai_messages
    ]
    oai_requests_cached = [
        model_api.call_cached(
            oai_models,
            prompt=message,
            max_tokens=200,
            n=1,
            print_prompt_and_response=True,
        )
        for message in oai_messages
    ]
    # answer = await asyncio.gather(
    #     *anthropic_requests, *oai_chat_requests, *oai_chat_requests_cached, *oai_requests, *oai_requests_cached
    # )
    # answer = await asyncio.gather(*oai_chat_requests_cached, *oai_requests_cached)

    costs = defaultdict(int)
    for responses in answer:
        for response in responses:
            costs[response.model_id] += response.cost

    print("-" * 80)
    print("Costs:")
    for model_id, cost in costs.items():
        print(f"{model_id}: ${cost}")

    # redis_wrapper.wait()
    return answer


# if __name__ == "__main__":
#     asyncio.run(demo())
