import os
from pathlib import Path
import shutil
import subprocess


def run_redis_in_background():
    assert "REDIS_READER_PORT" in os.environ

    redis_port = os.environ["REDIS_READER_PORT"]

    os.makedirs("./redis_dir", exist_ok=True)
    shutil.copy(Path(__file__).parent.parent / "redis.conf", "./redis_dir/redis.conf")

    log_file = "./log.txt"

    process = subprocess.Popen(
        ["redis-server", f"--port {redis_port}"],
        stdout=open(log_file, "a"),
        stderr=open(log_file, "a"),
        cwd="./redis_dir",
    )

    return process
