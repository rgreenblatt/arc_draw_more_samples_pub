import os
from pathlib import Path
import shutil
import subprocess


def run_redis_in_background():
    assert "REDIS_READER_PORT" in os.environ

    redis_port = os.environ["REDIS_READER_PORT"]

    os.makedirs("./redis_dir", exist_ok=True)
    shutil.copy(Path(__file__).parent.parent / "redis.conf", "./redis_dir/redis.conf")

    # cd ./redis_dir && redis-server --port $REDIS_READER_PORT &> log.txt &
    shell_cmd = f"redis-server --port {redis_port} &> ./log.txt &; disown %"
    subprocess.run(["bash", "-c", shell_cmd], cwd="./redis_dir", check=True)
