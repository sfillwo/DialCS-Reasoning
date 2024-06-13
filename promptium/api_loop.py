
import asyncio
import collections
import itertools
import time
import typing as T



class ApiLoop:
    def __init__(self, wait_policy=None):
        self.calls = collections.deque()
        self.awaiting = {}
        self.events = asyncio.Queue()
        self.loop = asyncio.get_event_loop()
        self.wait_policy = wait_policy

    def __call__(self, *coroutines):
        self.calls.extend(itertools.chain.from_iterable(
            (coro,) if asyncio.iscoroutine(coro) else coro
            for coro in coroutines
        ))
        self.awaiting.update(dict.fromkeys(self.calls))
        self.events.put_nowait(('call',))
        return self

    def __aiter__(self):
        return self

    async def _call(self):
        if not self.calls:
            return
        coroutine = self.calls.popleft()
        if self.wait_policy:
            wait = self.wait_policy.wait()
            if asyncio.iscoroutine(wait):
                await wait
        if hasattr(coroutine, 'get_wait_time'):
            wait_time = coroutine.get_wait_time()
            if wait_time:
                await asyncio.sleep(wait_time)
        if self.calls:
            self.events.put_nowait(('call', self.calls))
        if asyncio.iscoroutine(coroutine):
            result = await coroutine
        else:
            result = coroutine
        self.awaiting[coroutine] = result
        self.events.put_nowait(('reply', coroutine, result))
        return result

    async def __anext__(self):
        while self.awaiting:
            event, *value = await self.events.get()
            if event == 'call':
                asyncio.create_task(self._call())
            elif event == 'reply':
                coroutine, result = value
                self.awaiting.pop(coroutine)
                return result
        raise StopAsyncIteration

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = self.loop.run_until_complete(self.__anext__())
            return result
        except StopAsyncIteration:
            raise StopIteration





if __name__ == '__main__':

    import random

    async def api(s):
        chars = 'abcdefghijklmnopqrstuvwxyz'
        last_char = s[-1]
        char_pos = chars.index(last_char)
        next_chars = [chars[(char_pos + i + 1) % len(chars)] for i in range(3)]
        waiting_time = random.randint(1, 10)
        print(f'   API call with {s}...')
        await asyncio.sleep(waiting_time)
        result = [s + next_char for next_char in next_chars]
        print(f'   API call with {s} got reply {result} in {waiting_time}s')
        return result


    class MyWaitPolicy:
        def __init__(self, time):
            self.time = time
        async def wait(self):
            await asyncio.sleep(self.time)


    def main():
        loop = ApiLoop(wait_policy=MyWaitPolicy(1))
        ti = time.perf_counter()
        for result in loop(api('a'), api('b'), api('c')):
            print(result)
            loop(api('d'))
        tf = time.perf_counter()
        print(f'batch done in {tf - ti:.2f} seconds')


    # asyncio.run(amain())
    main()













