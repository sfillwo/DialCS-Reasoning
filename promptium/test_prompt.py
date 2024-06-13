
import pytest
import time
import asyncio

from promptium.prompt import Prompt, prompt

"""
account = OpenAiAccount()

chatGPT = GPT(account, 'gpt-3.5-turbo')

@prompt(chatGPT)
def my_prompt...
"""


def llm(prompt):
    time.sleep(2)
    return f'{prompt}!'

async def allm(prompt):
    await asyncio.sleep(2)
    return f'{prompt}!'


def test_pure_prompt_generation():
    p = Prompt(
        llm,
        template='Write a list of {n} animals',
    )
    assert p(n=10) == 'Write a list of 10 animals!'


def test_parsed_prompt_generation():
    p = Prompt(
        llm,
        template='squirrel, duck, rabbit, {animal}',
        parse=lambda gen: gen.split(', ')
    )
    assert p(animal='tiger') == ['squirrel', 'duck', 'rabbit', 'tiger!']


def test_pure_fn_prompt():
    @prompt(llm)
    def my_prompt(x: int, y: str):
        """
        Write a list of {x} {y}s
        """
    assert my_prompt(x=10, y='animal') == 'Write a list of 10 animals!'


def test_parsed_fn_prompt():
    @prompt(llm)
    def my_prompt(x: int=None, y: str=None, gen=None):
        """
        squirrel, duck, rabbit, {y}
        """
        return gen.split(', ')
    assert my_prompt(y='tiger') == ['squirrel', 'duck', 'rabbit', 'tiger!']


def test_delegate_fn_prompt():
    @prompt(llm)
    def my_prompt(x: int=None, y: str=None, llm:Prompt =None) -> list[str]:
        """
        Write a list of {x} {y}s
        """
        gen = llm(x=x + 1, y=y)
        return gen.split()
    assert my_prompt(x=10, y='animal') == ['Write', 'a', 'list', 'of', '11', 'animals!']


def test_async_pure_fn_prompt():
    @prompt(allm)
    def my_prompt(x: int, y: str):
        """
        Write a list of {x} {y}s
        """
    assert asyncio.run(my_prompt(x=10, y='animal')) == 'Write a list of 10 animals!'


def test_async_parsed_fn_prompt():
    @prompt(allm)
    def my_prompt(x: int=None, y: str=None, gen=None):
        """
        squirrel, duck, rabbit, {y}
        """
        return gen.split(', ')
    assert asyncio.run(my_prompt(y='tiger')) == ['squirrel', 'duck', 'rabbit', 'tiger!']


def test_async_delegate_fn_prompt():
    @prompt(allm)
    async def my_prompt(x: int=None, y: str=None, llm:Prompt =None) -> list[str]:
        """
        Write a list of {x} {y}s
        """
        gen = await llm(x=x + 1, y=y)
        return gen.split()
    assert asyncio.run(my_prompt(x=10, y='animal')) == ['Write', 'a', 'list', 'of', '11', 'animals!']