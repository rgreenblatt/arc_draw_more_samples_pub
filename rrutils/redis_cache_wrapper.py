import asyncio
import atexit
import hashlib
import json
import os
import random
from collections import deque
from typing import Optional

import attrs
import redis.asyncio as redis
from tqdm.asyncio import tqdm_asyncio


def get_digest(data):
    return hashlib.md5(json.dumps(data).encode()).hexdigest()


def cache_key(input_data, func_key):
    # Concatenate the function key with the MD5 hash of the JSON-serialized input data
    return f"{func_key}:{get_digest(input_data)}"


def get_client_reader(reader_port: int, writer_port: int):
    if reader_port == writer_port:
        redis_client = redis.Redis(host="localhost", port=reader_port)
        return redis_client, redis_client
    else:
        redis_client = redis.Redis(host="localhost", port=writer_port)
        redis_reader = redis.Redis(host="localhost", port=reader_port)
        return redis_client, redis_reader


def get_default_port():
    return int(os.environ.get("REDIS_READER_PORT", 6377))


def get_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        return asyncio.new_event_loop()


@attrs.define
class RedisWrapper:
    """
    # connect to the remote server Redis instance with
    ssh -i ~/.ssh/rr_dev.pem exx@64.255.46.66 -fNT -L 6377:localhost:6377

    # set up a local read replica with
    redis-server --port 6380 --slaveof 0.0.0.0 6377
    """

    port: int = attrs.field(default=get_default_port())
    batch_size: int = attrs.field(default=2000)
    batch_time: float = attrs.field(default=0.2)
    queue: deque = attrs.field(init=False, factory=deque)
    client: redis.Redis = attrs.field(init=False)
    loop: asyncio.AbstractEventLoop = attrs.field(init=False)
    has_items: asyncio.Event = attrs.field(init=False, factory=asyncio.Event)
    lock = asyncio.Lock()
    maximum_run_per_pipeline: int = (
        256  # this isn't reallly a serious bottleneck, I think the bottleneck is python code speed...
    )

    def __attrs_post_init__(self):
        self.client = redis.Redis(port=self.port)
        self.loop = get_event_loop()

    @classmethod
    def singleton(cls, port: Optional[int] = None):
        if not hasattr(cls, "_instance"):
            if port is None:
                port = get_default_port()
            cls._instance = cls(port=port)
        return cls._instance

    async def enqueue(self, operation, *args):
        future = self.loop.create_future()
        self.queue.append((operation, args, future))
        async with self.lock:
            if future.done():
                return future.result()
            await self.flush()

        assert future.done()
        return future.result()

    async def read(self, key_str, converter=None):
        key = f"json_{key_str}"
        value = await self.enqueue("GET", key)
        if value:
            value = value.decode("utf-8")

            if converter:
                return converter(value)
            else:
                return json.loads(value)
        else:
            return None

    async def lrange(self, idx, start, end):
        key = f"list_{idx}"
        values = await self.enqueue("LRANGE", key, start, end)

        return [json.loads(value) for value in values] if values else []

    async def rpush(self, idx, *values):
        key = f"list_{idx}"
        await self.enqueue("RPUSH", key, *[json.dumps(value) for value in values])

    async def write(self, key_str, value):
        key = f"json_{key_str}"
        return await self.enqueue("MSET", {key: json.dumps(value)})

    async def clear(self, idx):
        key = f"list_{idx}"
        await self.enqueue("DELETE", key)

    async def flush(self):
        if not self.queue:
            return

        pipeline = self.client.pipeline()
        futures = []
        mset_futures = []
        mset_dict = {}

        while self.queue and len(futures) < self.maximum_run_per_pipeline:
            operation, args, future = self.queue.popleft()
            if operation == "MSET":
                mset_futures.append(future)
                (arg,) = args
                mset_dict.update(arg)
            else:
                getattr(pipeline, operation.lower())(*args)
                futures.append(future)

        if len(mset_dict) > 0:
            pipeline.mset(mset_dict)

        results = await pipeline.execute()
        assert len(results) == len(futures) + (len(mset_dict) > 0)

        if len(mset_dict) > 0:
            mset_result = results[-1]
            results = results[:-1]
            for future in mset_futures:
                future.set_result(mset_result)

        for future, result in zip(futures, results):
            future.set_result(result)


async def test_all_funcs(i):
    client = RedisWrapper.singleton()

    await asyncio.sleep(random.random())

    # test read/write
    key = f"readwrite_{i}"
    write_read_val = {"hello": f"world_{i}"}
    await client.write(key, write_read_val)
    read_val = await client.read(key)
    assert read_val == write_read_val, f"read_val: {read_val}, write_read_val: {write_read_val}"

    await asyncio.sleep(random.random())
    # test rpush and lrange
    key = f"rpushlrange{i}"
    await client.rpush(key, "one", "two", "three")
    lrange_result = await client.lrange(key, 0, -1)
    assert len({"one", "two", "three"} - set(lrange_result)) == 0
    print("test passed!")


async def run_big_tests():
    await tqdm_asyncio.gather(*[test_all_funcs(i) for i in range(50000)])


if __name__ == "__main__":
    asyncio.run(run_big_tests())
