from __future__ import annotations

from enum import Enum, auto
from typing import Any, Literal, Optional, Protocol, Union

import attrs

import rrutils.optional as op

PRINT_COLORS = {"user": "cyan", "system": "magenta", "assistant": "light_green"}


class ContextLengthExceeded(Exception):
    original_exception: Exception

    def __init__(self, original_exception: Exception):
        super().__init__(str(original_exception))
        self.original_exception = original_exception


class StopReason(Enum):
    MAX_TOKENS = auto()
    STOP_SEQUENCE = auto()
    WEIRD = auto()

    @classmethod
    def factory(cls, stop_reason: Union[str, StopReason]) -> StopReason:
        """
        Parses the openai and anthropic stop reasons into a StopReason enum.
        """
        if isinstance(stop_reason, StopReason):
            return stop_reason

        stop_reason = stop_reason.lower()
        if stop_reason in ["max_tokens", "length"]:
            return cls.MAX_TOKENS
        elif stop_reason in ["stop_sequence", "stop"]:
            return cls.STOP_SEQUENCE
        elif stop_reason in ["weird"]:
            return cls.WEIRD
        raise ValueError(f"Invalid stop reason: {stop_reason}")

    @classmethod
    def from_anthropic(cls, stop_reason: Literal["end_turn", "max_tokens", "stop_sequence"]) -> StopReason:
        if stop_reason in ["max_tokens"]:
            return cls.MAX_TOKENS
        elif stop_reason in ["stop_sequence", "end_turn"]:
            # We could have a different one for stop_sequence vs end_turn
            return cls.STOP_SEQUENCE

        raise ValueError(f"Invalid stop reason: {stop_reason}")

    def __repr__(self):
        return self.name


@attrs.frozen
class TokenUsage:
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens


@attrs.frozen
class LLMResponse:
    model_id: str
    completion: str
    stop_reason: StopReason = attrs.field(converter=StopReason.factory)
    cost: float
    logprobs: Optional[list[dict[str, float]]] = None
    token_usage: Optional[TokenUsage] = None

    def as_dict(self):
        res = attrs.asdict(self)
        res["stop_reason"] = self.stop_reason.name
        return res

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        token_usage = op.map(d.get("token_usage"), lambda x: TokenUsage(**x))
        # d["stop_reason"] = StopReason[d["stop_reason"]]
        return cls(**{**d, "token_usage": token_usage})


class ModelAPIProtocol(Protocol):
    async def __call__(
        self, model_ids: list[str], prompt, print_prompt_and_response: bool, max_attempts: int, **kwargs
    ) -> list[LLMResponse]:
        raise NotImplementedError
