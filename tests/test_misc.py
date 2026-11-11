import asyncio

import pytest

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


async def test_disabled_notif_is_noop():
    notif = Notif(NotifyConf(enabled=False))
    assert notif.notif is None
    await notif.notify_change(50)
    await notif.notify_err(RuntimeError("x"))
