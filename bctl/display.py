import os
import logging
import asyncio
import aiofiles as aiof
import aiofiles.os as aios
from os import R_OK, W_OK
from abc import abstractmethod, ABC
from enum import StrEnum, auto
from typing import TypeVar
from asyncio import Task
from pathlib import Path
from logging import Logger
from .common import run_cmd, wait_and_reraise
from .config import Conf, OffsetType, SimConf
from .exceptions import ExitableErr, FatalErr


class DisplayType(StrEnum):
    INTERNAL = auto()
    EXTERNAL = auto()
    SIM = auto()


class BackendType(StrEnum):
    DDCUTIL = auto()  # note this can only be backend for external displays
    RAW = auto()
    BRILLO = auto()
    BRIGHTNESSCTL = auto()
    SIM = auto()


class Display(ABC):
    def __init__(self, id: str, dt: DisplayType, bt: BackendType, conf: Conf) -> None:
        self.id: str = id
        self.type: DisplayType = dt
        self.backend: BackendType = bt
        self.conf: Conf = conf
        self.name: str = "UNKNOWN"  # for ddcutil detect, it's the 'Monitor:' value
        self.raw_brightness: int = -1  # raw, i.e. potentially not a percentage
        self.max_brightness: int = -1
        self.logger: Logger = logging.getLogger(f"{type(self).__name__}.{self.id}")
        self.offset: int = 0  # percent
        self.eoffset: int = 0  # effective offset, percent

    def _init(self) -> None:
        for crit, offset in self.conf.offset.offsets.items():
            match crit:
                case "internal":
                    if self.type == DisplayType.INTERNAL:
                        self.offset = offset
                        break
                case "external":
                    if self.type == DisplayType.EXTERNAL:
                        self.offset = offset
                        break
                case "any":
                    self.offset = offset
                    break
                case _:
                    prefix = "model:"
                    if crit.startswith(prefix):
                        if crit[len(prefix):] == self.name:
                            self.offset = offset
                            break
                    else:
                        raise FatalErr(f"misconfigured offset criteria [{crit}]")

        if self.raw_brightness < 0:
            raise FatalErr(f"[{self.id}] raw_brightness appears to be uninitialized: {self.raw_brightness}")
        if self.max_brightness <= 0:
            raise FatalErr(f"[{self.id}] max_brightness appears to be uninitialized: {self.max_brightness}")
        if self.max_brightness < self.raw_brightness:
            raise FatalErr("max_brightness cannot be smaller than raw_brightness")

        self.logger.debug(
            f"  -> initialized {self.type} display [{self.id}], offset {self.offset}"
        )

    @abstractmethod
    async def _set_brightness(self, value: int) -> None:
        pass

    async def set_brightness(self, value: int) -> int:
        # print(f"  orig input value: {value}")
        orig_value = value
        if value < 0:
            value = 0
        elif value > 100:
            value = 100

        if self.offset != 0:
            if self.offset < 0:
                if self.conf.offset.type == OffsetType.SOFT and orig_value > 100:
                    target = min(100, orig_value + self.eoffset)
                else:
                    target = max(0, value + self.offset)
            else:  # offset > 0
                if self.conf.offset.type == OffsetType.SOFT and orig_value < 0:
                    target = max(0, orig_value + self.eoffset)
                else:
                    target = min(100, value + self.offset)
            self.eoffset = target - value
            # print(f"   -> target: {target}, eoffset: {self.eoffset}")
            value = target

        value = round(value / 100 * self.max_brightness)  # convert to raw

        if value != self.raw_brightness:
            self.logger.debug(
                f"setting display [{self.id}] brightness to {value} ({round(value / self.max_brightness * 100)}%)..."
            )
            await self._set_brightness(value)
            self.raw_brightness = value
        return self._get_brightness(False, False)

    async def adjust_brightness(self, delta: int) -> int:
        return await self.set_brightness(self._get_brightness(False, True) + delta)

    def get_brightness(self, raw: bool = False, offset_normalized: bool = False) -> int:
        self.logger.debug(f"getting display [{self.id}] brightness")
        return self._get_brightness(raw, offset_normalized)

    def _get_brightness(self, raw: bool, offset_normalized: bool) -> int:
        return (
            self.raw_brightness - (round(self.eoffset / 100 * self.max_brightness) if offset_normalized else 0)
            if raw
            else round(self.raw_brightness / self.max_brightness * 100) - (self.eoffset if offset_normalized else 0)
        )


# display baseclass that's not backed by ddcutil
class NonDDCDisplay(Display, ABC):
    def __init__(self, id: str, conf: Conf, bt: BackendType) -> None:
        if id.startswith("ddcci"):  # e.g. ddcci11
            dt = DisplayType.EXTERNAL
        else:
            dt = DisplayType.INTERNAL
        super().__init__(id, dt, bt, conf)


class SimulatedDisplay(Display):
    sim: SimConf

    def __init__(self, id: str, conf: Conf) -> None:
        super().__init__(id, DisplayType.SIM, BackendType.SIM, conf)
        self.sim = self.conf.sim

    async def init(self, initial_brightness: int) -> None:
        await asyncio.sleep(3)
        if self.sim.failmode == "i":
            raise ExitableErr(
                f"error initializing [{self.id}]", exit_code=self.sim.exit_code
            )
        self.raw_brightness = initial_brightness
        self.max_brightness = 100
        super()._init()

    async def _set_brightness(self, value: int) -> None:
        await asyncio.sleep(self.sim.wait_sec)
        if self.sim.failmode == "s":
            raise ExitableErr(
                f"error setting [{self.id}] brightness to {value}",
                exit_code=self.sim.exit_code,
            )


# for ddcutil performance, see https://github.com/rockowitz/ddcutil/discussions/393
class DDCDisplay(Display):
    bus: str  # string representation of this display's i2c bus number

    def __init__(self, id: str, conf: Conf) -> None:
        super().__init__(id, DisplayType.EXTERNAL, BackendType.DDCUTIL, conf)

    async def init(self) -> None:
        assert self.bus.isdigit(), f"[{self.id}] bus is invalid: [{self.bus}]"
        cmd = [
            "ddcutil",
            "--brief",
            "getvcp",
            self.conf.ddcutil_brightness_feature,
            "--bus",
            self.bus,
        ]
        out, err, code = await run_cmd(cmd, throw_on_err=True, logger=self.logger)
        out = out.split()
        assert len(out) == 5, f"{cmd} output unexpected: {out}"
        self.raw_brightness = int(out[-2])
        self.max_brightness = int(out[-1])
        super()._init()

    async def _set_brightness(self, value: int) -> None:
        await self._set_vcp_feature([self.conf.ddcutil_brightness_feature, str(value)])

    async def _set_vcp_feature(self, args: list[str]) -> None:
        await run_cmd(
            ["ddcutil"]
            + self.conf.ddcutil_svcp_flags
            + ["setvcp"]
            + args
            + ["--bus", self.bus],
            throw_on_err=True,
            logger=self.logger,
        )

    async def _get_vcp_feature(self, args: list[str]) -> str:
        out, err, code = await run_cmd(
            ["ddcutil"]
            + self.conf.ddcutil_gvcp_flags
            + ["getvcp"]
            + args
            + ["--bus", self.bus],
            throw_on_err=True,
            logger=self.logger,
        )
        return out


class BCTLDisplay(NonDDCDisplay):
    def __init__(self, bctl_out: str, conf: Conf) -> None:
        out = bctl_out.split(",")
        assert len(out) == 5, f"unexpected brightnessctl list output: [{bctl_out}]"
        self.raw_brightness = int(out[2])
        self.max_brightness = int(out[4])
        super().__init__(out[0], conf, BackendType.BRIGHTNESSCTL)

    async def init(self) -> None:
        super()._init()

    async def _set_brightness(self, value: int) -> None:
        await run_cmd(
            ["brightnessctl", "--quiet", "-d", self.id, "set", str(value)],
            throw_on_err=True,
            logger=self.logger,
        )


class BrilloDisplay(NonDDCDisplay):
    def __init__(self, id: str, conf: Conf) -> None:
        super().__init__(id, conf, BackendType.BRILLO)

    async def init(self) -> None:
        futures: list[Task[int]] = [
            asyncio.create_task(self._get_device_attr("b")),  # current brightness
            asyncio.create_task(self._get_device_attr("m")),  # max
        ]
        await wait_and_reraise(futures)
        self.raw_brightness = futures[0].result()
        self.max_brightness = futures[1].result()
        super()._init()

    async def _get_device_attr(self, attr: str) -> int:
        out, err, code = await run_cmd(
            ["brillo", "-s", self.id, f"-rlG{attr}"],
            throw_on_err=True,
            logger=self.logger,
        )
        return int(out)

    async def _set_brightness(self, value: int) -> None:
        await run_cmd(
            ["brillo", "-s", self.id, "-rlS", str(value)],
            throw_on_err=True,
            logger=self.logger,
        )


class RawDisplay(NonDDCDisplay):
    device_dir: str
    brightness_f: Path

    def __init__(self, device_dir: str, conf: Conf) -> None:
        super().__init__(os.path.basename(device_dir), conf, BackendType.RAW)
        self.device_dir = device_dir  # caller needs to verify it exists!

    async def init(self) -> None:
        self.brightness_f = Path(self.device_dir, "brightness")

        if not (
            await aios.path.isfile(self.brightness_f)
            and await aios.access(self.brightness_f, R_OK)
            and await aios.access(self.brightness_f, W_OK)
        ):
            raise FatalErr(f"[{self.brightness_f}] is not a file w/ RW perms")

        # self.raw_brightness = int(self.brightness_f.read_text().strip())  # non-async
        self.raw_brightness = await self._read_int(self.brightness_f)

        max_brightness_f = Path(self.device_dir, "max_brightness")
        if await aios.path.isfile(max_brightness_f) and await aios.access(
            max_brightness_f, R_OK
        ):
            # self.max_brightness = int(max_brightness_f.read_text().strip())  # non-async
            self.max_brightness = await self._read_int(max_brightness_f)
        super()._init()

    async def _read_int(self, file: Path) -> int:
        async with aiof.open(file, mode="r") as f:
            return int((await f.read()).strip())

    async def _set_brightness(self, value: int) -> None:
        async with aiof.open(self.brightness_f, mode="w") as f:
            await f.write(str(value))


TNonDDCDisplay = TypeVar("TNonDDCDisplay", bound=NonDDCDisplay)
TDisplay = TypeVar("TDisplay", bound=Display)
