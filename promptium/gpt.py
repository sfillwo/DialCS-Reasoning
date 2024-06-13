
from __future__ import annotations
import dataclasses as dc
import collections as coll
import pathlib
import time
import traceback
import sys
import itertools as it
import openai
import ezpyz as ez
from promptium.prompt import prompt
import typing as T


@dc.dataclass
class OpenAiAccount:
    key: str = None
    organization: str = None
    key_path: ez.filelike = pathlib.Path.home()/'.keys'/'openai'
    input_tokens_used: coll.defaultdict[str, int] = None
    output_tokens_used: coll.defaultdict[str, int] = None

    def __post_init__(self):
        if self.key is None and self.key_path is not None:
            key_file = ez.File(self.key_path)
            lines = [line.strip() for line in key_file.read().splitlines() if line.strip()]
            if len(lines) > 1:
                self.organization = lines[0]
                self.key = lines[1]
            else:
                self.key = lines[0]
        if self.input_tokens_used is None:
            self.input_tokens_used = coll.defaultdict(lambda: 0)
        if self.output_tokens_used is None:
            self.output_tokens_used = coll.defaultdict(lambda: 0)



@dc.dataclass
class GPT:
    account: OpenAiAccount
    model: str
    temperature: float = 1.0
    frequency_penalty: float = 0.0
    history: list[str] = None
    max_tokens: int|None = None
    catching_exceptions: tuple[T.Type[Exception], ...] = (Exception,)
    display_errors: bool = True

    def __post_init__(self):
        if self.history is None:
            self.history = []

    def prompt(self):
        return prompt(self)
    
    def format(self, prompt):
        history = [dict(role='user', content=prompt)]
        for speaker, turn in zip(it.cycle(['assistant', 'user']), reversed(self.history)):
            history.append(dict(role=speaker, content=turn))
        history.reverse()
        return history

    def __call__(self, prompt):
        openai.api_key = self.account.key
        openai.organization = self.account.organization
        history = self.format(prompt)
        try:
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=history,
                temperature=self.temperature,
                frequency_penalty=self.frequency_penalty,
                max_tokens=self.max_tokens,
            )
        except self.catching_exceptions as e:
            if self.display_errors:
                print(traceback.format_exc(), file=sys.stderr)
            return None
        else:
            input_tokens_used = completion.usage.prompt_tokens
            output_tokens_used = completion.usage.completion_tokens
            self.account.input_tokens_used[self.model] += input_tokens_used
            self.account.output_tokens_used[self.model] += output_tokens_used
            return completion.choices[0].message.content

    async def call(self, prompt):
        openai.api_key = self.account.key
        openai.organization = self.account.organization
        history = self.format(prompt)
        try:
            completion = await openai.Completion.acreate(
                model=self.model,
                messages=history,
                temperature=self.temperature,
                frequency_penalty=self.frequency_penalty,
                max_tokens=self.max_tokens,
            )
        except self.catching_exceptions as e:
            if self.display_errors:
                traceback.print_exc()
            return None
        else:
            input_tokens_used = completion.usage.prompt_tokens
            output_tokens_used = completion.usage.completion_tokens
            self.account.input_tokens_used[self.model] += input_tokens_used
            self.account.output_tokens_used[self.model] += output_tokens_used
            return completion.choices[0].text

