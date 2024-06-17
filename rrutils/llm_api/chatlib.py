import copy
import logging
import re
from typing import Optional

import attrs
import bs4
import tiktoken

from rrutils.llm_api.llm import ModelAPI
from rrutils.llm_api.openai_llm import count_chat_tokens
from rrutils.redis_cache_wrapper import get_digest

smart_models_strs = ["gpt-4", "gpt-4-0613"]
dumb_models_strs = ["gpt-3.5-turbo", "gpt-3.5-turbo-0613"]
dumb_models_long_strs = ["gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613"]
dumb_instruct_model_strs = ["gpt-3.5-turbo-instruct"]

tokenizer = tiktoken.encoding_for_model("gpt-4")


COLORS = {
    "system": "\033[94m",
    "user": "\033[92m",
    "assistant": "\033[96m",
    "gpt4": "\033[91m",
    "turbo": "\033[95m",
    "long_turbo": "\033[95m",
    "model": "\033[91m",
}


def print_messages(
    messages: list[dict[str, str]], hide_tags: set[str] = set(), file=None, color: bool = True, color_all: bool = True
):
    for message in messages:
        content = message["content"]
        for t in hide_tags:
            content = hide_xml(content, t)
        color_start = COLORS[message["role"]] if color else ""
        color_end = "\033[0m" if color else ""
        if color_all:
            s = f"{color_start}{message['role']}:\n{content}{color_end}"
        else:
            s = f"{color_start}{message['role']}{color_end}:\n{content}"
        print(s, file=file)


@attrs.define
class Chat:
    model_api: ModelAPI
    name: str = "root"
    echo: bool = False
    color_all: bool = True
    default_deterministic: bool = True
    messages: list[dict[str, str]] = attrs.field(factory=list)
    current_role: Optional[str] = None
    children: dict[str, "Chat"] = attrs.field(factory=dict)

    def hash(self):
        return get_digest([[sorted(x.items()) for x in self.messages], self.current_role])

    def say(self, role, content, silence=False, strip=True):
        if self.echo and not silence:
            # TODO: support color_all properly!!
            if role != self.current_role:
                print(f"{COLORS[role]}{role}\033[0m")
            else:
                print()
            print(content)
            # print(f"{role}: {content}")
        else:
            logging.info(f"{role}: {content}")
        if self.current_role == role:
            self.messages[-1]["content"] += "\n\n" + content
        else:
            self.messages.append({"role": role, "content": content})
            self.current_role = role

        if strip:
            self.messages[-1]["content"] = self.messages[-1]["content"].strip()

    def system_say(self, *content):
        self.say("system", " ".join(content))

    def user_say(self, *content, xml_wrap=None):
        if xml_wrap:
            self.say("user", f"<{xml_wrap}>\n{' '.join(content)}\n</{xml_wrap}>")
        else:
            self.say("user", " ".join(content))

    def assistant_say(self, *content):
        self.say("assistant", " ".join(content))

    async def ask_models(
        self,
        models,
        resample=False,
        parse=False,
        color_type: Optional[str] = None,
        max_tokens: Optional[int] = None,
        t: float = 0.0,
        **kwargs,
    ):
        if color_type is None:
            if models == dumb_models_strs:
                color_type_v = "turbo"
            elif models == dumb_models_long_strs:
                color_type_v = "long_turbo"
            elif models == smart_models_strs:
                color_type_v = "gpt4"
            else:
                color_type_v = "model"
        else:
            color_type_v = color_type
        if self.echo:
            print(f"{COLORS[color_type_v]}{color_type_v}\033[0m")
        res = (
            await self.model_api.call_cached(
                models,
                self.messages,
                max_tokens=max_tokens,
                compute_fresh=resample,
                n=1,
                t=t,
                deterministic=self.default_deterministic,
                **kwargs,
            )
        )[0]
        completion = res.completion

        self.say("assistant", completion, silence=not self.echo)
        if self.echo:
            print()
        # if ":bug:" in completion:
        #     self.user_say("Why do you think there's a bug?")
        #     resp = await self.ask_gpt4()
        #     raise ValueError("Model claims there's a bug: " + resp)
        if parse:
            return bs4.BeautifulSoup(completion, "html.parser")
        else:
            return completion

    async def ask_gpt4(self, resample=False, parse=False, max_tokens: Optional[int] = None, t: float = 0.0, **kwargs):
        return await self.ask_models(
            smart_models_strs, resample=resample, parse=parse, color_type="gpt4", max_tokens=max_tokens, t=t, **kwargs
        )

    async def ask_turbo(
        self,
        resample=False,
        parse=False,
        max_tokens: Optional[int] = None,
        t: float = 0.0,
        use_long: bool = False,
        **kwargs,
    ):
        return await self.ask_models(
            dumb_models_long_strs if use_long else dumb_models_strs,
            resample=resample,
            parse=parse,
            color_type="turbo",
            max_tokens=max_tokens,
            t=t,
            **kwargs,
        )

    def print(self, hide_tags: set[str] = set(), file=None, color: bool = True):
        print_messages(self.messages, hide_tags, file, color, color_all=self.color_all)

    def subchat(self, name: Optional[str] = None) -> "Chat":
        if name is None:
            name = str(len(self.children))

        assert name not in self.children, f"Chat {name} already exists"
        new_chat = attrs.evolve(
            self, name=name, messages=copy.deepcopy(self.messages), current_role=str(self.current_role), children=dict()
        )
        self.children[name] = new_chat
        return new_chat

    def message_tree(self, prefix=""):
        name = f"{prefix}/{self.name}"
        tree = {name: self.messages}
        for child in self.children.values():
            tree.update(child.message_tree(prefix=name))
        return tree

    def current_tokens(self):
        return count_chat_tokens(self.messages)


def hide_xml(text, tag):
    pattern = f"<{tag}>(.*?)</{tag}>"
    return re.sub(pattern, lambda x: f"<{tag}>\n...\n</{tag}>", text, flags=re.DOTALL)


def extract_xml(text, tag):
    pattern = f"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.DOTALL)

    if len(matches) == 1:
        assert "```" not in matches[0]
        return matches[0]
    elif len(matches) > 1:
        raise ValueError(f"Error: Multiple instances found:\n{text}")
    else:
        raise ValueError(f"Error: No instance found:\n{text}")


def extract_code_from_md(text):
    pattern = r"```(?:[a-zA-Z0-9\+]+)?\n(.*?)\n```"
    return re.findall(pattern, text, re.DOTALL)


def extract_code(text):
    try:
        return extract_xml(text, "code")
    except ValueError:
        return extract_code_from_md(text)[0]
