import asyncio
import logging
import os
import re
from typing import Optional, Protocol

import attrs
from anthropic import AsyncAnthropic
from anthropic.types.completion import Completion as AnthropicCompletion
from anthropic.types import Message as AnthropicMessage, MessageParam as AnthropicMessageParam
from termcolor import cprint

from rrutils.llm_api.base_llm import PRINT_COLORS, LLMResponse, ModelAPIProtocol, StopReason, TokenUsage
import rrutils.optional as op

ANTHROPIC_MODELS = {"claude-instant-1", "claude-2", "claude-v1.3"}

ANTHROPIC_NEW_MODELS = {
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-2.1",
    "claude-2.0",
    "claude-instant-1.2",
}


def count_tokens(prompt: str) -> int:
    return len(prompt.split())


def price_per_token(model_id: str) -> tuple[float, float]:
    """
    Returns the (input token, output token) price for the given model id.
    """

    if model_id.startswith("claude-3-opus"):
        return 15 / 1_000_000, 75 / 1_000_000
    elif model_id.startswith("claude-3-sonnet"):
        return 3 / 1_000_000, 15 / 1_000_000
    elif model_id.startswith("claude-3-haiku"):
        return 0.25 / 1_000_000, 1.25 / 1_000_000

    # TODO
    return 0.0, 0.0


@attrs.define()
class AnthropicChatModel(ModelAPIProtocol):
    num_threads: int
    client: AsyncAnthropic = attrs.field(init=False, default=attrs.Factory(AsyncAnthropic))
    available_requests: asyncio.BoundedSemaphore = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.available_requests = asyncio.BoundedSemaphore(self.num_threads)

    async def __call__(
        self, model_ids: list[str], prompt: str, print_prompt_and_response: bool, max_attempts: int, **kwargs
    ) -> list[LLMResponse]:
        assert len(model_ids) == 1, "Anthropic implementation only supports one model at a time."
        model_id = model_ids[0]

        response: Optional[AnthropicCompletion] = None
        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    with open("/tmp/latest_anthropic_semaphore.txt", "w") as f:
                        f.write(str(self.num_threads - self.available_requests._value))
                    response = await self.client.completions.create(prompt=prompt, model=model_id, **kwargs)

            except Exception as e:
                logging.warn(f"{model_id} encountered API error: {str(e)}.\nRetrying now. (Attempt {i})")
                await asyncio.sleep(min(1.5**i, 60.0 * 2))
            else:
                break

        if response is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        with open("/tmp/latest_anthropic_semaphore.txt", "w") as f:
            f.write(str(self.num_threads - self.available_requests._value))

        num_context_tokens, num_completion_tokens = await asyncio.gather(
            self.client.count_tokens(prompt), self.client.count_tokens(response.completion)
        )
        context_token_cost, completion_token_cost = price_per_token(model_id)
        cost = num_context_tokens * context_token_cost + num_completion_tokens * completion_token_cost

        llm_response = LLMResponse(
            model_id=model_id, completion=response.completion, stop_reason=response.stop_reason, cost=cost
        )

        if print_prompt_and_response:
            pattern = r"\n\n((Human: |Assistant: ).*)"
            for match in re.finditer(pattern, prompt):
                role = match.group(2).removesuffix(": ").lower()
                role = {"human": "user"}.get(role, role)
                cprint(match.group(1), PRINT_COLORS[role])
            cprint(f"Response ({llm_response.model_id}):", "white")
            cprint(f"{llm_response.completion}", PRINT_COLORS["assistant"], attrs=["bold"])
            print()

        return [llm_response]


AnthropicNewChatPrompt = list[AnthropicMessageParam]


@attrs.define()
class AnthropicNewChatModel(ModelAPIProtocol):
    num_threads: int
    client: AsyncAnthropic = attrs.field(init=False, default=attrs.Factory(AsyncAnthropic))
    available_requests: asyncio.BoundedSemaphore = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.available_requests = asyncio.BoundedSemaphore(self.num_threads)

    async def __call__(
        self,
        model_ids: list[str],
        messages: AnthropicNewChatPrompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        stop_sequences: list[str] = [],
        add_stop_sequence_to_completion: bool = False,
        **kwargs,
    ) -> list[LLMResponse]:
        assert len(model_ids) == 1, "Anthropic implementation only supports one model at a time."
        model_id = model_ids[0]

        if print_prompt_and_response:
            raise NotImplementedError("Printing prompts and responses is not yet supported for the new chat model.")

        response: Optional[AnthropicMessage] = None
        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    with open("/tmp/latest_anthropic_semaphore.txt", "w") as f:
                        f.write(str(self.num_threads - self.available_requests._value))
                    response = await self.client.messages.create(
                        model=model_id, messages=messages, stop_sequences=stop_sequences, **kwargs
                    )

            except Exception as e:
                logging.warn(f"{model_id} encountered API error: {str(e)}.\nRetrying now. (Attempt {i})")
                await asyncio.sleep(min(1.5**i, 60.0 * 2))
            else:
                break

        if response is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        with open("/tmp/latest_anthropic_semaphore.txt", "w") as f:
            f.write(str(self.num_threads - self.available_requests._value))

        num_context_tokens, num_completion_tokens = response.usage.input_tokens, response.usage.output_tokens
        context_token_cost, completion_token_cost = price_per_token(model_id)
        cost = num_context_tokens * context_token_cost + num_completion_tokens * completion_token_cost

        con = response.content
        if len(con) == 0:
            completion_text = ""  # TODO, kinda hacky...
        else:
            assert len(con) == 1, (
                con,
                response.usage,
                response,
            )  # TODO: it's possible for this list to be empty if output tokens == 1
            completion_text = con[0].text

        if add_stop_sequence_to_completion:
            assert len(stop_sequences) > 0
            if response.stop_sequence is not None:
                assert isinstance(response.stop_sequence, str)
                completion_text += response.stop_sequence

        llm_response = LLMResponse(
            model_id=model_id,
            completion=completion_text,
            stop_reason=StopReason.from_anthropic(op.unwrap(response.stop_reason)),
            cost=cost,
            token_usage=TokenUsage(input_tokens=num_context_tokens, output_tokens=num_completion_tokens),
        )

        return [llm_response]
