import asyncio
import logging

import pytest

from bctl import common
from bctl.common import assert_cmd_exist, run_cmd, same_values, wait_and_reraise
from bctl.exceptions import CmdErr, ExitableErr, FatalErr

LOGGER = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "values, expected",
    [([1], True), ([3, 3, 3], True), ([3, 3, 4], False), (["a", "b"], False)],
)
def test_same_values(values, expected):
    assert same_values(values) is expected


def test_socket_path_is_under_runtime_dir():
    assert common.SOCKET_PATH.startswith(common.RUNTIME_PATH)
    assert common.SOCKET_PATH.endswith("/bctl/bctld-ipc.sock")


async def test_run_cmd_returns_stdout_stderr_and_code():
    out, err, code = await run_cmd(["sh", "-c", "echo out; echo err >&2"], LOGGER)
    assert (out, err, code) == ("out\n", "err\n", 0)


async def test_run_cmd_splits_string_command():
    out, _, code = await run_cmd("echo a b", LOGGER)
    assert (out, code) == ("a b\n", 0)


async def test_run_cmd_nonzero_exit_without_throw():
    _, _, code = await run_cmd(["sh", "-c", "exit 3"], LOGGER)
    assert code == 3


async def test_run_cmd_nonzero_exit_throws_cmderr():
    with pytest.raises(CmdErr) as exc:
        await run_cmd(["sh", "-c", "echo boom >&2; exit 4"], LOGGER, throw_on_err=True)
    assert exc.value.code == 4
    assert exc.value.stderr == "boom\n"


def test_assert_cmd_exist():
    assert_cmd_exist("sh")
    with pytest.raises(FatalErr):
        assert_cmd_exist("bctl-surely-nonexistent-binary")


async def test_wait_and_reraise_all_succeed():
    tasks = [asyncio.create_task(asyncio.sleep(0, result=i)) for i in range(3)]
    await wait_and_reraise(tasks)
    assert [t.result() for t in tasks] == [0, 1, 2]


async def test_wait_and_reraise_propagates_and_cancels_pending():
    async def boom():
        raise ValueError("boom")

    slow = asyncio.create_task(asyncio.sleep(10))
    with pytest.raises(ValueError, match="boom"):
        await wait_and_reraise([asyncio.create_task(boom()), slow])
    assert slow.cancelled()


async def test_wait_and_reraise_timeout_raises_silent_exitable(monkeypatch):
    orig_wait = asyncio.wait

    def short_wait(tasks, timeout, return_when):
        return orig_wait(tasks, timeout=0.05, return_when=return_when)

    monkeypatch.setattr(common.asyncio, "wait", short_wait)
    slow = asyncio.create_task(asyncio.sleep(10))
    with pytest.raises(ExitableErr, match="1 / 2 tasks timed out") as exc:
        await wait_and_reraise([asyncio.create_task(asyncio.sleep(0)), slow])
    assert exc.value.notify is False
    assert slow.cancelled()
