from __future__ import annotations
import asyncio
import dataclasses as dc
import inspect
import pathlib
import textwrap
import ezpyz as ez
import typing as T

factory = lambda x: dc.field(default_factory=x)

class GenerationStore(ez.Store[dict[str, str]]):
    def __init__(self, path:ez.filelike):
        super().__init__(data=None, path=path, format=ez.format.JSON)


@dc.dataclass
class Prompt:
    llm: T.Callable[..., T.Any]
    template: str|None = None
    prompt: str|None = None
    prefix: str|None = None
    display: bool = False
    parse: T.Callable[..., T.Any]|None = None
    delegate: T.Optional[T.Callable] = None
    store: ez.filelike|None = 'llm_cache'
    __call__: T.Callable[..., T.Any] = None

    def __post_init__(self):
        if self.store and not isinstance(self.store, GenerationStore):
            self.store:GenerationStore = GenerationStore(self.store) # noqa
        if asyncio.iscoroutinefunction(self.llm):
            self.__call__ = self.call_async
        else:
            self.__call__ = self.call

    def fill(self, *args, **kwargs):
        self.prompt = self.template.format(*args, **kwargs)
        return self.prompt

    def gen(self, prompt=None, prefix=None, display=None):
        llm = dc.replace(self, **ez.undefault(prompt=prompt, prefix=prefix, display=display))
        instance_key = str(llm.llm.format(prompt))
        if llm.store and instance_key in llm.store.data:
            print('Retrieving from cache...')
            generation = llm.store.data[instance_key]
        else:
            generation = llm.llm(llm.prompt)
            if generation is not None and llm.store:
                llm.store.data[instance_key] = generation
                llm.store.save()
        if llm.display:
            print(llm.prompt, generation, '\n', sep=' ')
        if generation is not None and llm.prefix:
            generation = llm.prefix + generation
        return generation

    async def gen_async(self, prompt=None, prefix=None, display=None):
        llm = dc.replace(self, **ez.undefault(prompt=prompt, prefix=prefix, display=display))
        instance_key = str(llm.llm.format(prompt))
        if llm.store and instance_key in llm.store.data:
            generation = llm.store.data[instance_key]
        else:
            generation = await llm.llm(llm.prompt)
            if generation is not None and llm.store:
                llm.store.data[instance_key] = generation
        if llm.display:
            print(llm.prompt, generation, '\n', sep=' ')
        if generation is not None and llm.prefix:
            generation = llm.prefix + generation
        return generation

    def call(self, *args, display=None, **kwargs):
        if display is None:
            display = self.display
        if self.delegate:
            llm = dc.replace(self, delegate=None, **ez.undefault(display=display))
            return self.delegate(*args, llm=llm, **kwargs)
        else:
            llm = dc.replace(self, display=display)
            prompt = llm.fill(*args, **kwargs)
            generation = llm.gen(prompt, llm.prefix)
            if llm.parse:
                return llm.parse(gen=generation)
            else:
                return generation

    async def call_async(self, *args, display=None, **kwargs):
        if display is None:
            display = self.display
        if self.delegate:
            llm = dc.replace(self, delegate=None, **ez.undefault(display=display))
            return await self.delegate(*args, llm=llm, **kwargs)
        else:
            llm = dc.replace(self, display=display)
            prompt = llm.fill(*args, **kwargs)
            generation = await llm.gen_async(prompt, llm.prefix)
            if llm.parse:
                return llm.parse(gen=generation)
            else:
                return generation

    def __call__(self, *args, **kwargs): # noqa
        return self.__call__(*args, **kwargs)


F = T.TypeVar('F', bound=T.Callable[..., T.Any])

def prompt(
    llm:T.Callable[[str], str]|T.Callable[[str], T.Awaitable[str]] = None
):
    def decorator(fn:F) -> F | Prompt:
        if llm is None:
            llm_name = ''
        else:
            llm_name = llm.__name__ if hasattr(llm, '__name__') else type(llm).__name__
        name = fn.__name__
        parameter_names = list(inspect.signature(fn).parameters)
        delegate = fn if 'llm' in parameter_names else None
        parser = fn if 'gen' in parameter_names else None
        lines = fn.__doc__.split('\n')
        if not lines[0].strip():
            if not lines[-1].strip():
                lines = lines[1:-1]
            else:
                lines = lines[1:]
        last_documentation_index = len(lines)
        for i, trailing_line in enumerate(reversed(lines)):
            if not (
                trailing_line.strip().startswith(':param') or
                trailing_line.strip().startswith(':return:')
            ):
                if i > 0:
                    last_documentation_index = len(lines) - i - 1
                break
        lines = lines[:last_documentation_index]
        template = textwrap.dedent('\n'.join(lines))
        if '||' in template:
            template, prefix = template.split('||', 1)
            template = template + prefix
        else:
            prefix = None
        store = pathlib.Path('generations')/llm_name/f'{name}.json'
        p = Prompt(
            llm=llm,
            template=template,
            prefix=prefix,
            delegate=delegate,
            parse=parser,
            store=store
        )
        return p
    return decorator






























