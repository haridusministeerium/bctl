from typing import Any

from dbus_fast import Message, MessageType, Variant
from dbus_fast.aio import MessageBus

from bctl.config import NotifyConf, NotifyIconConf

APP_NAME = "bctld"
# spec @ https://specifications.freedesktop.org/notification-spec/latest/
NOTIF_BUS = "org.freedesktop.Notifications"
NOTIF_PATH = "/org/freedesktop/Notifications"
NOTIF_SIGNATURE = "susssasa{sv}i"  # app_name, replaces_id, app_icon, summary, body, actions, hints, expire_timeout


class Notif:
    def __init__(self, conf: NotifyConf) -> None:
        self.conf: NotifyConf = conf
        self.icon_conf: NotifyIconConf = conf.icon
        self._bus: MessageBus | None = None

    # icon spec @ https://specifications.freedesktop.org/icon-theme-spec/latest/
    def _get_notif_icon(self, val: int) -> str:
        if val <= 10:
            icon = self.icon_conf.brightness_off
        elif val <= 30:
            icon = self.icon_conf.brightness_low
        elif val <= 55:
            icon = self.icon_conf.brightness_medium
        elif val <= 85:
            icon = self.icon_conf.brightness_high
        else:
            icon = self.icon_conf.brightness_full

        i_root = self.icon_conf.root_dir
        if icon and i_root and not i_root.endswith("/"):
            i_root += "/"
        return i_root + icon

    async def _notify(
        self, summary: str, body: str, icon: str, hints: dict[str, Variant], timeout_ms: int
    ) -> None:
        if self._bus is None or not self._bus.connected:
            self._bus = await MessageBus().connect()
        args: list[Any] = [APP_NAME, 0, icon, summary, body, [], hints, timeout_ms]
        reply = await self._bus.call(
            Message(
                destination=NOTIF_BUS,
                path=NOTIF_PATH,
                interface=NOTIF_BUS,
                member="Notify",
                signature=NOTIF_SIGNATURE,
                body=args,
            )
        )
        if reply.message_type is MessageType.ERROR:
            raise RuntimeError(f"desktop notification failed: {reply.error_name}: {reply.body}")

    async def notify_err(self, err: Exception) -> None:
        if not self.conf.enabled or not self.conf.on_fatal_err:
            return
        await self._notify("bctl", str(err), self.icon_conf.error, {}, 0)  # timeout 0 = never expire

    # note as per https://wiki.archlinux.org/title/Desktop_notifications#Replace_previous_notification
    # 'string:x-canonical-private-synchronous:' hint is for replacing existing notification
    # the same way referencing it by ID does
    async def notify_change(self, val: int) -> None:
        if not self.conf.enabled:
            return
        hints = {
            "value": Variant("i", val),
            "x-canonical-private-synchronous": Variant("s", "brightness_bctld"),
        }
        await self._notify(str(val), "", self._get_notif_icon(val), hints, self.conf.timeout_ms)
