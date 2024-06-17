from queue import Empty
import os
import random
import multiprocessing
import asyncio
from typing import Optional, Any
from tqdm.asyncio import tqdm_asyncio

import attrs


def get_default_num_workers():
    MIN_SET_ASIDE_CORES = 4
    FRAC_CORES = 1 / 4

    total_cores = os.cpu_count()
    assert total_cores is not None
    worker_cores = min(FRAC_CORES * total_cores, total_cores - MIN_SET_ASIDE_CORES)

    max_workers = max(1, int(worker_cores))
    return max_workers


def run_function_from_queues(
    func, queue_in: multiprocessing.Queue, queue_out: multiprocessing.Queue
):
    limit_memory()

    # TODO: better exception handling, currently this just kills the process and then we wait for a timeout, but we could skip the timeout...
    while True:
        args, kwargs = queue_in.get(block=True)
        out = func(*args, **kwargs)
        queue_out.put(out)


def get_with_timeout(queue: multiprocessing.Queue, timeout=1.0):
    try:
        return queue.get(block=True, timeout=timeout)
    except Empty:
        raise asyncio.TimeoutError


@attrs.frozen
class TimeoutProcessItem:
    proc: multiprocessing.Process
    inp_queue: multiprocessing.Queue
    out_queue: multiprocessing.Queue

    async def call_with_timeout_fixed(
        self, *args, timeout: float = 1.0, default_on_timeout: Any = None, **kwargs
    ):
        self.inp_queue.put((args, kwargs))

        try:
            return (
                await asyncio.to_thread(
                    get_with_timeout, self.out_queue, timeout=timeout
                ),
                self,
            )
        except asyncio.TimeoutError:
            self.proc.terminate()
            self.proc.join()

            return default_on_timeout, None


def limit_memory():
    import resource

    # Set the memory limit in bytes
    memory_limit = 6 * 1024**3

    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, hard))


@attrs.define
class TimeoutProcessItemWrapper:
    item: Optional[TimeoutProcessItem] = None

    async def call_with_timeout_fixed(
        self,
        func,
        *args,
        timeout: float = 1.0,
        default_on_timeout: Any = None,
        **kwargs,
    ):
        if self.item is None:
            # only start a process if we had to kill a prior process!!!
            inp_queue = multiprocessing.Queue()
            out_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=run_function_from_queues, args=(func, inp_queue, out_queue)
            )
            process.start()
            self.item = TimeoutProcessItem(process, inp_queue, out_queue)

        out, new_item = await self.item.call_with_timeout_fixed(
            *args, timeout=timeout, default_on_timeout=default_on_timeout, **kwargs
        )
        self.item = new_item
        return out


async def run_func_with_timeout_with_resources(
    func,
    process_wrappers_with_locks: list[tuple[asyncio.Lock, TimeoutProcessItemWrapper]],
    *args,
    timeout: float = 1.0,
    default_on_timeout: Any = None,
    wrap_call: Optional[tuple[Any, Any]] = None,
    **kwargs,
):
    idxs = [
        i
        for i, (lock, _) in enumerate(process_wrappers_with_locks)
        if not lock.locked()
    ]  # hack to avoid unnecessary contention
    if len(idxs) == 0:
        idxs = list(range(len(process_wrappers_with_locks)))
    idx = random.choice(idxs)  # obviously suboptimal...

    lock, item = process_wrappers_with_locks[idx]
    async with lock:
        if wrap_call is not None:
            done, output = await wrap_call[0](*args, **kwargs)
            if done:
                return output

        out = await item.call_with_timeout_fixed(
            func,
            *args,
            timeout=timeout,
            default_on_timeout=default_on_timeout,
            **kwargs,
        )

        if wrap_call is not None:
            await wrap_call[1](out, *args, **kwargs)

        return out


async def map_fixed_func_with_timeout_with_resources(
    func,
    process_wrappers_with_locks: list[tuple[asyncio.Lock, TimeoutProcessItemWrapper]],
    arg_kwarg_items: list[tuple[tuple, dict]],
    timeout: float = 1.0,
    default_on_timeout: Any = None,
    wrap_call: Optional[tuple[Any, Any]] = None,  # this is useful for caching!
):
    # could allow for disabling tqdm here
    return await tqdm_asyncio.gather(
        *[
            run_func_with_timeout_with_resources(
                func,
                process_wrappers_with_locks,
                *args,
                timeout=timeout,
                default_on_timeout=default_on_timeout,
                wrap_call=wrap_call,
                **kwargs,
            )
            for args, kwargs in arg_kwarg_items
        ]
    )


async def map_fixed_func_with_timeout(
    func,
    arg_kwarg_items: list[tuple[tuple, dict]],
    n_workers: int = get_default_num_workers(),  # arbitrary default
    timeout: float = 1.0,
    default_on_timeout: Any = None,
    wrap_call: Optional[tuple[Any, Any]] = None,
):
    process_wrappers_with_locks = [
        (asyncio.Lock(), TimeoutProcessItemWrapper()) for _ in range(n_workers)
    ]
    return await map_fixed_func_with_timeout_with_resources(
        func,
        process_wrappers_with_locks,
        arg_kwarg_items,
        timeout=timeout,
        default_on_timeout=default_on_timeout,
        wrap_call=wrap_call,
    )
