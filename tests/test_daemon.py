import asyncio
import json
import os

import pytest

from bctl import daemon
from bctl.common import SOCKET_PATH, Opts
from bctl.config import Conf, GetStrategy, State
from bctl.display import DisplayType
from bctl.exceptions import CmdErr, FatalErr, PayloadErr

from conftest import FakeNotif


@pytest.fixture
def env(monkeypatch, conf, make_display):
    """Populate the daemon's module-level globals with fakes; returns them for assertions."""

    class Env:
        pass

    e = Env()
    e.conf = conf
    e.notif = FakeNotif()
    e.queue = asyncio.Queue()
    e.init_calls = 0
    e.internal = make_display("eDP", dt=DisplayType.INTERNAL)
    e.external = make_display("ext", dt=DisplayType.EXTERNAL)
    e.displays = [e.internal, e.external]

    async def fake_init_displays():
        e.init_calls += 1

    monkeypatch.setattr(daemon, "CONF", conf, raising=False)
    monkeypatch.setattr(daemon, "NOTIF", e.notif, raising=False)
    monkeypatch.setattr(daemon, "TASK_QUEUE", e.queue, raising=False)
    monkeypatch.setattr(daemon, "LOCK", asyncio.Lock(), raising=False)
    monkeypatch.setattr(daemon, "DISPLAYS", e.displays)
    monkeypatch.setattr(daemon, "init_displays", fake_init_displays)
    return e


def brightnesses(e) -> list[int]:
    return [d.get_brightness() for d in e.displays]


class TestExecuteTasks:
    async def test_batched_up_down_coalesce_into_single_write(self, env):
        await daemon.execute_tasks([["up", None], ["up", None], ["down", 3]])
        assert brightnesses(env) == [57, 57]
        assert env.internal.writes == [57]
        assert env.notif.changes == [57]
        assert env.conf.state.last_set_brightness == 57

    async def test_set_cancels_earlier_deltas(self, env):
        await daemon.execute_tasks([["delta", 0, 10], ["set", 0, 30], ["delta", 0, 5]])
        assert brightnesses(env) == [35, 35]

    async def test_explicit_up_amount(self, env):
        await daemon.execute_tasks([["up", 20]])
        assert brightnesses(env) == [70, 70]

    async def test_zero_delta_is_noop(self, env):
        await daemon.execute_tasks([["delta", 5], ["delta", -5]])
        assert env.internal.writes == []
        assert env.notif.changes == []

    async def test_no_notify_and_no_track(self, env):
        await daemon.execute_tasks([["set", Opts.NO_NOTIFY | Opts.NO_TRACK, 20]])
        assert brightnesses(env) == [20, 20]
        assert env.notif.changes == []
        assert env.conf.state.last_set_brightness == -1

    async def test_ignore_internal_only_touches_external(self, env):
        await daemon.execute_tasks([["set", Opts.IGNORE_INTERNAL, 80]])
        assert brightnesses(env) == [50, 80]

    @pytest.mark.parametrize("task", [["init"], ["init", 0, 0], ["init", 2, 0.01]])
    async def test_init_variants_reinit_displays(self, env, task):
        await daemon.execute_tasks([task])
        assert env.init_calls == 1

    async def test_sync_task(self, env):
        env.external.raw_brightness = 70
        await daemon.execute_tasks([["sync"]])
        assert await env.queue.get() == ["set", 0, 60]

    @pytest.mark.parametrize("task, code", [(["kill"], 0), (["kill", 3], 3)])
    async def test_kill(self, env, task, code):
        with pytest.raises(SystemExit) as exc:
            await daemon.execute_tasks([task])
        assert exc.value.code == code

    async def test_unknown_task_is_ignored(self, env):
        await daemon.execute_tasks([["bogus", 1]])
        assert env.internal.writes == []

    async def test_failed_write_triggers_reinit_and_is_swallowed(self, env):
        env.external.fail_with = CmdErr("ddcutil died", 1, "")
        await daemon.execute_tasks([["set", 0, 10]])
        assert env.init_calls > 0
        assert env.notif.changes == []


class TestSyncDisplays:
    @pytest.fixture(autouse=True)
    def differing(self, env):
        env.internal.raw_brightness = 40
        env.external.raw_brightness = 60

    async def queued(self, env):
        await daemon.sync_displays()
        return None if env.queue.empty() else env.queue.get_nowait()

    @pytest.mark.parametrize(
        "strategy, target",
        [
            (("mean",), 50),
            (("low",), 40),
            (("high",), 60),
            (("internal",), 40),
            (("external",), 60),
            (("id:ext",), 60),
            (("id:laptop",), 40),
            (("id:missing", "high"), 60),
        ],
    )
    async def test_strategies(self, env, strategy, target):
        env.conf.sync_strategy = strategy
        assert await self.queued(env) == ["set", 0, target]

    async def test_last_set_brightness_takes_precedence(self, env):
        env.conf.state.last_set_brightness = 70
        assert await self.queued(env) == ["set", 0, 70]

    async def test_no_matching_strategy_skips_sync(self, env):
        env.displays.remove(env.internal)
        env.displays.append(env.external.__class__("ext2", env.conf, raw=10))
        env.displays[-1]._init()
        env.conf.sync_strategy = ("internal",)
        assert await self.queued(env) is None

    async def test_misconfigured_strategy_is_fatal(self, env):
        env.conf.sync_strategy = ("bogus",)
        with pytest.raises(FatalErr):
            await daemon.sync_displays()

    async def test_already_in_sync_within_tolerance(self, env):
        env.external.raw_brightness = 41
        assert await self.queued(env) is None

    async def test_single_display_never_synced(self, env):
        env.displays.remove(env.external)
        assert await self.queued(env) is None

    async def test_post_set_syncs_when_enabled(self, env):
        env.conf.sync_brightness = True
        await daemon.post_set([40, 60], 0)
        assert await env.queue.get() == ["set", 0, 50]

    async def test_post_set_respects_no_sync(self, env):
        env.conf.sync_brightness = True
        await daemon.post_set([40, 60], Opts.NO_SYNC)
        assert env.queue.empty()


class TestGetBrightness:
    def test_uniform_value(self, env):
        assert daemon.get_brightness(0, False) == (50,)

    @pytest.mark.parametrize(
        "strategy, expected",
        [(GetStrategy.MEAN, 50), (GetStrategy.LOW, 40), (GetStrategy.HIGH, 60)],
    )
    def test_strategy_consolidates_differing_values(self, env, strategy, expected):
        env.internal.raw_brightness = 40
        env.external.raw_brightness = 60
        env.conf.get_strategy = strategy
        assert daemon.get_brightness(0, False) == (expected,)

    def test_get_all(self, env):
        env.external.raw_brightness = 60
        assert daemon.get_brightness(Opts.GET_ALL, []) == [("eDP", 50), ("ext", 60)]

    def test_by_display_name(self, env):
        assert daemon.get_brightness(0, ["laptop"]) == [("eDP", 50)]

    def test_filtered_to_nothing(self, env):
        assert daemon.get_brightness(0, ["nope"]) == []
        assert daemon.get_brightness(Opts.IGNORE_INTERNAL | Opts.IGNORE_EXTERNAL, []) == []


async def test_display_op_without_matching_displays_raises_payload_err(env):
    with pytest.raises(PayloadErr) as exc:
        await daemon.display_op(lambda d: d.set_brightness(1), lambda d: False)
    assert exc.value.payload[0] == 1


@pytest.mark.parametrize(
    "err, payload",
    [(PayloadErr("x", [1, "custom"]), [1, "custom"]), (ValueError("bad"), [1, "bad"])],
)
def test_handle_failure_after_retries(err, payload):
    assert daemon.handle_failure_after_retries(err) == payload


async def test_process_q_batches_within_consumption_window(env, monkeypatch):
    env.conf.msg_consumption_window_sec = 0.05
    batches: list[list[list]] = []

    async def record(tasks):
        batches.append(tasks)

    monkeypatch.setattr(daemon, "execute_tasks", record)
    consumer = asyncio.create_task(daemon.process_q())
    for _ in range(3):
        await env.queue.put(["up", None])
    await asyncio.sleep(0.2)
    await env.queue.put(["down", None])
    await asyncio.sleep(0.2)
    consumer.cancel()

    assert batches == [[["up", None]] * 3, [["down", None]]]


class TestSocketProtocol:
    @pytest.fixture
    async def server(self, env):
        err_fut = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(daemon.process_client_commands(err_fut))
        for _ in range(100):
            if os.path.exists(SOCKET_PATH):
                break
            await asyncio.sleep(0.01)
        yield env
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def send(self, cmd: list):
        reader, writer = await asyncio.open_unix_connection(SOCKET_PATH)
        writer.write(json.dumps(cmd).encode())
        writer.write_eof()
        data = await reader.read()
        writer.close()
        await writer.wait_closed()
        return json.loads(data) if data else None

    async def test_get(self, server):
        assert await self.send(["get"]) == [0, 50]

    async def test_get_all(self, server):
        assert await self.send(["get", Opts.GET_ALL, []]) == [0, ["eDP", 50], ["ext", 50]]

    async def test_get_unknown_display(self, server):
        assert await self.send(["get", 0, ["nope"]]) == [0]

    async def test_set_get(self, server):
        assert await self.send(["set_get", 0, 0, 0, 70]) == [0, ["eDP", 70], ["ext", 70]]
        assert server.notif.changes == [70]

    async def test_set_get_for(self, server):
        resp = await self.send(["set_get_for", 0, 0, 0, {"laptop": 30}])
        assert resp == [0, ["eDP", 30]]
        assert brightnesses(server) == [30, 50]

    async def test_set_for_async_replies_nothing(self, server):
        assert await self.send(["set_for_async", {"ext": 20}]) is None
        for _ in range(100):
            if server.external.get_brightness() == 20:
                break
            await asyncio.sleep(0.01)
        assert brightnesses(server) == [50, 20]

    async def test_getvcp_without_ddc_displays_fails(self, server):
        resp = await self.send(["getvcp", 0, 0, [], ["10"]])
        assert resp[0] == 1

    async def test_queued_command_gets_no_reply(self, server):
        assert await self.send(["up", None]) is None
        assert await server.queue.get() == ["up", None]


def test_validate_ext_deps_missing_binary(monkeypatch):
    conf = Conf(state=State())
    monkeypatch.setattr(daemon, "CONF", conf, raising=False)
    monkeypatch.setattr("bctl.common.shutil.which", lambda cmd: None)
    with pytest.raises(FatalErr, match="ddcutil"):
        daemon.validate_ext_deps()
