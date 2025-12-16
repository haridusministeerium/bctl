import glob
import os
from pathlib import Path
import aiofiles.os as aios
import re
import logging
import asyncio
from asyncio import Task
from logging import Logger
from bctl.common import (
    run_cmd,
    wait_and_reraise,
    Opts,
)
from typing import Callable
from collections.abc import Sequence
from bctl.display import (
    BackendType,
    DisplayType,
    Display,
    SimulatedDisplay,
    DDCDisplay,
    BCTLDisplay,
    BrilloDisplay,
    RawDisplay,
    TNonDDCDisplay,
    TDisplay,
    get_disp_filter,
)
from bctl.config import (
    Conf,
    MainDisplayCtl,
    InternalDisplayCtl,
)
from bctl.exceptions import (
    FatalErr,
    CmdErr,
)


LOGGER: Logger = logging.getLogger(__name__)


def nullify_offset(displays: Sequence[Display]) -> None:
    for d in displays:
        d.offset = 0
        d.eoffset = 0
    #conf.state.offsets = {}  # note offsets in state need to be reset as well


async def resolve_displays(conf: Conf) -> Sequence[Display]:
    if conf.sim:
        return await init_displays_sim(conf)

    LOGGER.debug("initing displays...")

    displays: Sequence[Display]
    match conf.main_display_ctl:
        case MainDisplayCtl.DDCUTIL:
            displays = await get_ddcutil_displays(conf)
        case MainDisplayCtl.RAW:
            displays = await get_raw_displays(conf)
        case MainDisplayCtl.BRIGHTNESSCTL:
            displays = await get_bctl_displays(conf)
        case MainDisplayCtl.BRILLO:
            displays = await get_brillo_displays(conf)

    filters: list[Callable[[Display], bool]] = []
    opts = 0
    if conf.ignore_internal_display:
        opts |= Opts.IGNORE_INTERNAL
    if conf.ignore_external_display:
        opts |= Opts.IGNORE_EXTERNAL
    if opts:
        filters.append(get_disp_filter(opts))

    if conf.ignored_displays:
        filters.append(lambda d: not bool(conf.ignored_displays & d.names))

    if filters:
        displays = list(filter(lambda d: all(f(d) for f in filters), displays))

    if len(list(filter(lambda d: d.type is DisplayType.INTERNAL, displays))) > 1:
        # TODO: shouldn't this exit fatally?
        raise RuntimeError("more than 1 laptop/internal displays found")

    if displays:
        futures: list[Task[None]] = [asyncio.create_task(d.init()) for d in displays]
        await wait_and_reraise(futures)

        # note offset nullification needs to happen _after_ displays have been init()'d:
        enabled_rule = conf.offset.enabled_if
        disabled_rule = conf.offset.disabled_if
        if ((enabled_rule and not eval(enabled_rule)) or
                (disabled_rule and eval(disabled_rule)) or
                (len(displays) == 1 and not conf.offset.enabled_if_single_display)):
            LOGGER.debug("disabling all offsets")
            nullify_offset(displays)

    LOGGER.debug(
        f"...initialized {len(displays)} display{'' if len(displays) == 1 else 's'}"
    )
    return displays


async def init_displays_sim(conf: Conf) -> Sequence[Display]:
    global DISPLAYS

    ndisplays: int = conf.sim.ndisplays

    LOGGER.debug(f"initing {ndisplays} simulated displays...")
    displays: list[SimulatedDisplay] = [
        SimulatedDisplay(f"sim-{i}", conf) for i in range(ndisplays)
    ]

    futures: list[Task[None]] = [asyncio.create_task(d.init()) for d in displays]
    await wait_and_reraise(futures)

    LOGGER.debug(
        f"...initialized {len(displays)} simulated display{'' if len(displays) == 1 else 's'}"
    )
    return displays


async def resolve_single_internal_display_raw(conf: Conf) -> RawDisplay:
    d = await get_raw_displays(conf)
    return _filter_internal_display(d, BackendType.RAW)


def _filter_by_backend_type(
    displays: list[TDisplay], bt: BackendType
) -> list[TDisplay]:
    return list(filter(lambda d: d.backend is bt, displays))


def _filter_by_display_type(
    displays: list[TDisplay], dt: DisplayType
) -> list[TDisplay]:
    return list(filter(lambda d: d.type is dt, displays))


def _filter_internal_display(
    disp: list[TNonDDCDisplay], provider: BackendType
) -> TNonDDCDisplay:
    displays: list[TNonDDCDisplay] = _filter_by_display_type(disp, DisplayType.INTERNAL)
    assert len(displays) == 1, (
        f"found {len(displays)} laptop/internal displays w/ {provider}, expected 1"
    )
    return displays[0]


async def resolve_single_internal_display_brillo(conf: Conf) -> BrilloDisplay:
    d = await get_brillo_displays(conf)
    return _filter_internal_display(d, BackendType.BRILLO)


async def resolve_single_internal_display_bctl(conf: Conf) -> BCTLDisplay:
    d = await get_bctl_displays(conf)
    return _filter_internal_display(d, BackendType.BRIGHTNESSCTL)


async def get_raw_displays(conf: Conf) -> list[RawDisplay]:
    device_dirs: list[str] = glob.glob(conf.raw_device_dir + "/*")
    assert len(device_dirs) > 0, "no backlight-capable raw devices found"

    return [
        RawDisplay(d, conf) for d in device_dirs if await aios.path.exists(d)
    ]  # exists() check to deal with dead symlinks


async def get_brillo_displays(conf: Conf) -> list[BrilloDisplay]:
    out, err, code = await run_cmd(["brillo", "-Ll"], LOGGER, throw_on_err=True)
    out = out.splitlines()
    assert len(out) > 0, "no backlight-capable devices found w/ brillo"

    return [
        BrilloDisplay(os.path.basename(i), conf)
        for i in out
        if await aios.path.exists(Path(conf.raw_device_dir, i))
    ]  # exists() check to deal with dead symlinks


async def get_bctl_displays(conf: Conf) -> list[BCTLDisplay]:
    cmd = ["brightnessctl", "--list", "--machine-readable", "--class=backlight"]
    out, err, code = await run_cmd(cmd, LOGGER, throw_on_err=True)
    out = out.splitlines()
    assert len(out) > 0, "no backlight-capable devices found w/ brightnessctl"

    return [
        BCTLDisplay(i, conf)
        for i in out
        if await aios.path.exists(Path(conf.raw_device_dir, i.split(",")[0]))
    ]  # exists() check to deal with dead symlinks


async def get_ddcutil_displays(conf: Conf) -> list[Display]:
    displays: list[Display] = []
    in_invalid_block = False
    d: list[str] = []
    out, err, code = await run_cmd(
        ["ddcutil", "--brief", "detect"], LOGGER, throw_on_err=False
    )
    if code != 0:
        # err string can be found in https://github.com/rockowitz/ddcutil/blob/master/src/app_ddcutil/main.c
        if err and "ddcutil requires module i2c" in err:
            raise FatalErr("ddcutil requires i2c-dev kernel module to be loaded")
        LOGGER.error(err)
        raise CmdErr(
            f"ddcutil failed to list/detect devices (exit code {code})", code, err
        )

    for line in out.splitlines():
        if not line:
            in_invalid_block = False
            if d:
                displays.append(DDCDisplay(d, conf))
                d = []  # reset
        elif in_invalid_block:  # try to detect laptop internal display
            # note matching against "eDP" in "DRM connector" line is not infallible, see https://github.com/rockowitz/ddcutil/issues/547#issuecomment-3253325547
            # expected line will be something like "   DRM connector:    card0-eDP-1"
            if re.fullmatch(
                r"\s+DRM connector:\s+[a-z0-9]+-eDP-\d+", line
            ):  # i.e. "is this a laptop display?"
                match conf.internal_display_ctl:
                    case InternalDisplayCtl.RAW:
                        displays.append(await resolve_single_internal_display_raw(conf))
                    case InternalDisplayCtl.BRIGHTNESSCTL:
                        displays.append(
                            await resolve_single_internal_display_bctl(conf)
                        )
                    case InternalDisplayCtl.BRILLO:
                        displays.append(
                            await resolve_single_internal_display_brillo(conf)
                        )
                in_invalid_block = False
        elif d or line.startswith("Display "):
            d.append(line.strip())
        elif line == "Invalid display" and not conf.ignore_internal_display:
            # start processing one of the 'Invalid display' blocks:
            in_invalid_block = True
    if d:  # sanity
        raise FatalErr("ddc display block parsing intiated but not finalized")
    return displays
