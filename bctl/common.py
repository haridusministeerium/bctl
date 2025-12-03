import os
import asyncio
import shutil
from asyncio import Task
from collections.abc import Sequence
from enum import IntFlag
from .exceptions import FatalErr, CmdErr


class Opts(IntFlag):
    NO_NOTIFY = 1
    NO_TRACK = 2
    IGNORE_EXTERNAL = 4
    IGNORE_INTERNAL = 8
    GET_INDIVIDUAL = 16
    GET_RAW = 32
    GET_NO_OFFSET_NORMALIZED = 64
    # NEXT_OPT = 128


def _runtime_path() -> str:
    xdg_dir = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    dir_path = xdg_dir + "/bctl"
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path


RUNTIME_PATH = _runtime_path()
SOCKET_PATH: str = f"{RUNTIME_PATH}/bctld-ipc.sock"


# input sequence cannot be empty!
def same_values(s: Sequence) -> bool:
    return s.count(s[0]) == len(s)


async def run_cmd(
    cmd: Sequence[str], throw_on_err=False, logger=None
) -> tuple[str, str, int | None]:
    if isinstance(cmd, str):
        cmd = cmd.split()

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        if logger:
            logger.error(f"{cmd} returned w/ {proc.returncode}")
        if throw_on_err:
            raise CmdErr(
                f"{cmd} returned w/ {proc.returncode}", proc.returncode, stderr.decode()
            )
    return stdout.decode(), stderr.decode(), proc.returncode


def assert_cmd_exist(cmd: str) -> None:
    if shutil.which(cmd) is None:
        raise FatalErr(f"external command [{cmd}] does not exist on our PATH")


# convenience method for waiting for futures' completion. it was created so any
# exceptions thrown in coroutines would be propagated up, and not swallowed.
# looks like task cancellation is the key for this, at least w/ return_when=asyncio.FIRST_EXCEPTION
async def wait_and_reraise(futures: Sequence[Task]) -> None:
    try:
        done, tasks_to_cancel = await asyncio.wait(
            futures, timeout=5, return_when=asyncio.FIRST_EXCEPTION
        )
    except asyncio.CancelledError:
        tasks_to_cancel = futures
        raise
    finally:
        for task in tasks_to_cancel:
            task.cancel()

    for task in done:
        if exc := task.exception():
            # print(f'exc type: {type(exc)}: {exc}')
            raise exc
