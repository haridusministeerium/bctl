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
        self.min_brightness: int = 0
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
        self.logger.debug(
            f"  -> initializing {self.type} display [{self.id}], offset {self.offset}..."
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
                    # TODO: unsure which one is better:
                    target = 100
                    # target = min(100, orig_value + self.offset)
                else:
                    target = max(0, value + self.offset)
            else:  # offset > 0
                if self.conf.offset.type == OffsetType.SOFT and orig_value < 0:
                    # TODO: unsure which one is better:
                    target = 0
                    # target = max(0, orig_value + self.offset)
                else:
                    target = min(100, value + self.offset)
            self.eoffset = target - value
            # print(f"   -> target: {target}, eoffset: {self.eoffset}")
            value = target

        value = round(value / 100 * self.max_brightness)  # convert to raw

        # TODO: this bound-checking is no longer needed due to checing at the top, right?:
        # if value < self.min_brightness:
            # value = self.min_brightness
        # elif value > self.max_brightness:
            # value = self.max_brightness

        if value != self.raw_brightness:
            self.logger.debug(
                f"setting display [{self.id}] brightness to {value} ({round(value / self.max_brightness * 100)}%)..."
            )
            await self._set_brightness(value)
            self.raw_brightness = value
        return self.get_brightness()

    async def adjust_brightness(self, delta: int) -> int:
        b: int = round(self.raw_brightness / self.max_brightness * 100)  # percentage
        additional_offset = 0

        if delta > 0:
            hard_limit = 100 - self.offset
            hard_limit2 = hard_limit * 2  # to allow other screens to reach 100%
            # if b == 100:
            if self.offset <= 0:
                additional_offset: int = max(self.offset - self.eoffset, -delta)
                if self.conf.offset.type == OffsetType.SOFT:
                    target: int = min(100, b + delta + additional_offset)
                    d = delta + additional_offset  # remove the "dead" buffer where only offset is increased
                    d = min(d, 100 - b)   # asked/input delta might be more than is available 'til max, hence min()
                    over_limit_d = min(0, (100 + self.offset) - (b + d))  # how much _over_ the limit to go
                else:  # OffsetType.HARD
                    target: int = min(100 + self.offset, b + delta + additional_offset)
                    over_limit_d = 0
                self.eoffset += additional_offset - over_limit_d
            else:  # offset > 0
                if self.offset and self.eoffset == 3 * self.offset:
                    return self.get_brightness()  # we've hit abolute limit at the high-end (that's only achievable with OffsetType.SOFT
                target = min(100, b + delta)
                additional_offset = min(self.offset - self.eoffset, delta)  # in case we're overextended on low end and need to give some offset back
                if additional_offset > 0:
                    print(f"     -> additional offset: {additional_offset}")
                    new_target_plus_offset = min(100, target + additional_offset)
                    # the actual _extra_ is what can be added back to the offset; note this causes effective delta to be more than asked delta:
                    self.eoffset += new_target_plus_offset - target
                    target = new_target_plus_offset
                # elif target > hard_limit2:
                    # return self.get_brightness()

                # delta = b
                # limit = target - b
                # limit = target - hard_limit
                # if limit > 0: # and :
                print(f"     -> target {target}")
                if target > hard_limit:
                    # e = target - self.offset + self.eoffset
                    e = target + self.eoffset
                    f = e - hard_limit
                    # if target == 100 and self.eoffset == self.offset:
                        # f -= self.offset
                    if target == 100:
                        self.eoffset += delta
                        # f -= self.offset
                    print(f"          e: {e}; f: {f}")
                    virtual_brightness = b + self.eoffset - self.offset
                    # f = virtual_brightness - hard_limit
                    print(
                        f"     -> virt_b: {virtual_brightness};   min({100 + 3 * self.offset}, {b + delta})"
                    )

                    d = min(100 + 3 * self.offset, b + delta)  # + self.eoffset)
                    #d = min(100 + 3 * self.offset, min(b + delta, b + self.offset)) # + self.eoffset)
                    # self.eoffset += d - 100
                    # self.eoffset += f
                    print(
                        f"     -> d = {d}, target {target} hard_limit {hard_limit} eoffset by {d - 100}"
                    )

        else:  # delta <= 0
            if self.offset >= 0:
                if self.offset:
                    if self.eoffset == 3 * self.offset:
                        additional_offset = max(self.offset - self.eoffset, -delta)
                    else:
                        additional_offset = min(self.offset - self.eoffset, -delta)

                if self.conf.offset.type == OffsetType.SOFT:
                    if self.offset and self.eoffset == 3 * self.offset:
                        target = min(100, b + delta + self.eoffset)# + additional_offset)  # <-- TODO: commented tail out in latest (Mon) edit!
                    else:
                        target = max(0, b + delta)# + additional_offset)  # <-- TODO: commented tail out in latest (Mon) edit!

                    d = delta + additional_offset  # remove the "dead" buffer where only offset is increased
                    d = max(d, 0 - b)   # asked/input delta might be more than is available 'til min, hence max()
                    over_limit_d = max(0, (0 + self.offset) - (b + d))  # how much _over_ the limit to go
                    # TODO: shouldn't additional_offset really be function of new $target?
                    # e.g. w/ offset 5, going from 5 to 0, our eoffset will be 5 += (additional_offset - over_limit_d) = (0 - 5) = -5
                    # e.g. w/ offset 5, going from 100 to 95, our eoffset will be 10 += (additional_offset - over_limit_d) = (-5 - 0) = -5
                    # or when we're fully extended towards high:
                    # TODO: following is wrong!!! we still want eoffset to be 15 += -5!
                    # e.g. w/ offset 5, going from 100 to 95, our eoffset will be 15 += (additional_offset - over_limit_d) = (-10 - 0) = -10
                    print(
                        f"     -> !additional offset: {additional_offset}, over_limit_d = {over_limit_d}; new eoffset will be {self.eoffset} += {additional_offset - over_limit_d}, target: {target}"
                    )
                else:  # OffsetType.HARD
                    target = max(0 + self.offset, b + delta + additional_offset)
                    over_limit_d = 0
                self.eoffset += additional_offset - over_limit_d
            else:  # offset < 0
                target = max(0, b + delta)
                additional_offset = max(self.offset - self.eoffset, delta)  # in case we're overextended on high end and need to give some offset back
                if additional_offset:
                    new_target_plus_offset = max(0, target + additional_offset)
                    # the actual _extra_ is what can be added back to the offset; note this causes effective delta to be more than asked delta:
                    self.eoffset += new_target_plus_offset - target
                    target = new_target_plus_offset

        target = round(target / 100 * self.max_brightness)  # convert to raw value
        self.logger.debug(
            f"adjusting display [{self.id}] brightness to {target} ({round(target / self.max_brightness * 100)}%), offset {self.offset}, eoffset {self.eoffset}..."
        )
        await self._set_brightness(target)
        self.raw_brightness = target
        return self.get_brightness()

    def get_brightness(self, raw: bool = False, offset_normalized: bool = False) -> int:
        if self.raw_brightness == -1:
            raise FatalErr(f"[{self.id}] appears to be uninitialized")
        self.logger.debug(f"getting display [{self.id}] brightness")

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
        super()._init()
        await asyncio.sleep(3)
        if self.sim.failmode == "i":
            raise ExitableErr(
                f"error initializing [{self.id}]", exit_code=self.sim.exit_code
            )
        self.raw_brightness = initial_brightness
        self.max_brightness = 100

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
        super()._init()
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
        assert self.max_brightness >= self.raw_brightness, (
            "max_brightness cannot be smaller than raw_brightness"
        )

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
        assert self.max_brightness >= self.raw_brightness, (
            "max_brightness cannot be smaller than raw_brightness"
        )
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
        super()._init()

        futures: list[Task[int]] = [
            asyncio.create_task(self._get_device_attr("b")),  # current brightness
            asyncio.create_task(self._get_device_attr("m")),  # max
            asyncio.create_task(self._get_device_attr("c")),  # min
        ]
        await wait_and_reraise(futures)
        self.raw_brightness = futures[0].result()
        self.max_brightness = futures[1].result()
        self.min_brightness = futures[2].result()
        assert self.max_brightness >= self.raw_brightness, (
            "max_brightness cannot be smaller than raw_brightness"
        )

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
        super()._init()

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
            # assert self.max_brightness >= self.raw_brightness, "max_brightness cannot be smaller than raw_brightness"

    async def _read_int(self, file: Path) -> int:
        async with aiof.open(file, mode="r") as f:
            return int((await f.read()).strip())

    async def _set_brightness(self, value: int) -> None:
        async with aiof.open(self.brightness_f, mode="w") as f:
            await f.write(str(value))


TNonDDCDisplay = TypeVar("TNonDDCDisplay", bound=NonDDCDisplay)
TDisplay = TypeVar("TDisplay", bound=Display)
