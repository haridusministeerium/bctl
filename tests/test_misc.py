import asyncio
from types import SimpleNamespace

import pytest
from dbus_fast import Message, MessageType, Variant

from bctl import notify as notify_mod
from bctl.config import NotifyConf, NotifyIconConf
from bctl.debouncer import Debouncer
from bctl.notify import Notif


async def test_debouncer_only_runs_last_call():
    calls: list[int] = []

    async def record(v):
        calls.append(v)

    debounced = Debouncer(delay=0.05)
    for i in range(3):
        await debounced(record, i)
    await asyncio.sleep(0.15)
    assert calls == [2]

    await debounced(record, 9)
    await asyncio.sleep(0.15)
    assert calls == [2, 9]


@pytest.mark.parametrize(
    "val, icon",
    [
        (0, "off"),
        (10, "off"),
        (11, "low"),
        (30, "low"),
        (55, "medium"),
        (85, "high"),
        (86, "full"),
        (100, "full"),
    ],
)
def test_notif_icon_thresholds(val, icon):
    notif = Notif(NotifyConf(enabled=False, icon=NotifyIconConf(root_dir="/icons")))
    assert notif._get_notif_icon(val) == f"/icons/notification-display-brightness-{icon}.svg"


def test_notif_icon_single_icon_via_root_dir():
    icons = {f"brightness_{lvl}": "" for lvl in ("full", "high", "medium", "low", "off")}
    notif = Notif(NotifyConf(enabled=False, icon=NotifyIconConf(root_dir="/x/one.svg", **icons)))
    assert notif._get_notif_icon(50) == "/x/one.svg"


class FakeBus:
    def __init__(self, reply_type: MessageType = MessageType.METHOD_RETURN) -> None:
        self.connected = False
        self.sent: list[Message] = []
        self.reply_type = reply_type

    async def connect(self) -> "FakeBus":
        self.connected = True
        return self

    async def call(self, msg: Message) -> SimpleNamespace:
        self.sent.append(msg)
        return SimpleNamespace(message_type=self.reply_type, error_name="org.x.Err", body=["boom"])


@pytest.fixture
def bus(monkeypatch):
    b = FakeBus()
    monkeypatch.setattr(notify_mod, "MessageBus", lambda: b)
    return b


async def test_disabled_notif_is_noop(bus):
    notif = Notif(NotifyConf(enabled=False))
    await notif.notify_change(50)
    await notif.notify_err(RuntimeError("x"))
    assert bus.sent == []


async def test_notify_change_sends_notify_call(bus):
    notif = Notif(NotifyConf(timeout_ms=1234, icon=NotifyIconConf(root_dir="/icons")))
    await notif.notify_change(70)
    await notif.notify_change(71)

    assert len(bus.sent) == 2  # bus connection is reused
    msg = bus.sent[0]
    assert (msg.destination, msg.path, msg.interface, msg.member, msg.signature) == (
        "org.freedesktop.Notifications",
        "/org/freedesktop/Notifications",
        "org.freedesktop.Notifications",
        "Notify",
        "susssasa{sv}i",
    )
    app, replaces_id, icon, summary, body, actions, hints, timeout = msg.body
    assert (app, replaces_id, summary, body, actions, timeout) == ("bctld", 0, "70", "", [], 1234)
    assert icon == "/icons/notification-display-brightness-high.svg"
    assert hints["value"] == Variant("i", 70)
    assert hints["x-canonical-private-synchronous"] == Variant("s", "brightness_bctld")


async def test_notify_err_never_expires(bus):
    await Notif(NotifyConf()).notify_err(RuntimeError("bad thing"))
    _, _, icon, summary, body, _, hints, timeout = bus.sent[0].body
    assert (icon, summary, body, hints, timeout) == ("gtk-dialog-error", "bctl", "bad thing", {}, 0)


async def test_notify_err_respects_on_fatal_err(bus):
    await Notif(NotifyConf(on_fatal_err=False)).notify_err(RuntimeError("x"))
    assert bus.sent == []


async def test_notify_error_reply_raises(bus):
    bus.reply_type = MessageType.ERROR
    with pytest.raises(RuntimeError, match="org.x.Err"):
        await Notif(NotifyConf()).notify_change(50)
