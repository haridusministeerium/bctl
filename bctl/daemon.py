import asyncio
import traceback
import signal
import os
import sys
import re
import json
import logging
from functools import partial
import glob
from retry_deco import RetryAsync, OnErrOpts
from logging import Logger
from types import TracebackType
from asyncio import AbstractEventLoop, Task, Queue, Lock, Event
from typing import NoReturn, Callable, Coroutine, Any, List, Sequence
from tendo import singleton
from pathlib import Path
from statistics import fmean
from contextlib import suppress
import aiofiles.os as aios
from .debouncer import Debouncer
from .udev_monitor import monitor_udev_events
from .display import (
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
)
from .common import (
    run_cmd,
    same_values,
    assert_cmd_exist,
    wait_and_reraise,
    Opts,
    SOCKET_PATH,
)
from .config import (
    Conf,
    SimConf,
    load_config,
    write_state,
    unix_time_now,
    MainDisplayCtl,
    InternalDisplayCtl,
    GetStrategy,
)
from .exceptions import ExitableErr, FatalErr, PayloadErr, CmdErr, RetriableException
from .notify import Notif

DISPLAYS: Sequence[Display] = []
TASK_QUEUE: Queue[list]
LOCK: Lock
CONF: Conf
LOGGER: Logger = logging.getLogger(__name__)
NOTIF: Notif
LAST_INIT_TIME: int = 0


def validate_ext_deps() -> None:
    requirements = [CONF.main_display_ctl, CONF.internal_display_ctl]
    for dep in ["ddcutil", "brillo", "brightnessctl"]:
        if dep in requirements:
            assert_cmd_exist(dep)


async def init_displays() -> None:
    global DISPLAYS
    global LAST_INIT_TIME
    DISPLAYS = []  # immediately reset old state

    if CONF.sim:
        return await init_displays_sim(CONF.sim)

    LOGGER.debug("initing displays...")
    ignore_internal: bool = CONF.ignore_internal_display

    displays: Sequence[Display]
    match CONF.main_display_ctl:
        case MainDisplayCtl.DDCUTIL:
            displays = await get_ddcutil_displays(ignore_internal)
        case MainDisplayCtl.RAW:
            displays = await get_raw_displays()
        case MainDisplayCtl.BRIGHTNESSCTL:
            displays = await get_bctl_displays()
        case MainDisplayCtl.BRILLO:
            displays = await get_brillo_displays()

    if ignore_internal:
        displays = list(filter(lambda d: d.type != DisplayType.INTERNAL, displays))
    if CONF.ignore_external_display:
        displays = list(filter(lambda d: d.type != DisplayType.EXTERNAL, displays))

    ignored_displays = CONF.ignored_displays
    if ignored_displays:
        displays = list(
            filter(
                lambda d: d.id not in ignored_displays
                and d.name not in ignored_displays,
                displays,
            )
        )

    if len(list(filter(lambda d: d.type == DisplayType.INTERNAL, displays))) > 1:
        # TODO: shouldn't this exit fatally?
        raise RuntimeError("more than 1 laptop/internal displays found")

    if displays:
        futures: List[Task[None]] = [asyncio.create_task(d.init()) for d in displays]
        await wait_and_reraise(futures)
        for d in displays:
            for o in CONF.state.offsets:
                id, name = o[0]
                if (name and d.name == name) or d.id == id:
                    d.eoffset = o[1]
                    break

    DISPLAYS = displays
    enabled_rule = CONF.offset.enabled_if
    if enabled_rule and not eval(enabled_rule):
        LOGGER.debug(f"[{enabled_rule}] evaluated false, disabling offsets...")
        for d in DISPLAYS:
            d.offset = 0
            d.eoffset = 0
    disabled_rule = CONF.offset.disabled_if
    if disabled_rule and eval(disabled_rule):
        LOGGER.debug(f"[{disabled_rule}] evaluated true, disabling offsets...")
        for d in DISPLAYS:
            d.offset = 0
            d.eoffset = 0

    LOGGER.debug(
        f"...initialized {len(displays)} display{'' if len(displays) == 1 else 's'}"
    )

    if CONF.sync_brightness:
        await sync_displays()
    # TODO: is resetting last_set_brightness reasonable here? even when a new display
    #       gets connected, doesn't it still remain our last requested brightness target?
    elif not same_values([d.get_brightness(offset_normalized=True) for d in DISPLAYS]):
        CONF.state.last_set_brightness = -1  # reset, as potentially newly added display could have a different value

    LAST_INIT_TIME = unix_time_now()


async def sync_displays() -> None:
    if len(DISPLAYS) <= 1:
        return
    values: List[int] = [d.get_brightness(offset_normalized=True) for d in DISPLAYS]
    if same_values(values):
        return

    target: int = CONF.state.last_set_brightness
    if target == -1:  # i.e. we haven't explicitly set it to anything yet
        d: Display | None = None
        strat = CONF.sync_strategy
        for s in strat:
            match s:
                case "mean":
                    target = int(fmean(values))
                    break
                case "low":
                    target = min(values)
                    break
                case "high":
                    target = max(values)
                    break
                case "internal":
                    d = next((d for d in DISPLAYS if d.type == DisplayType.INTERNAL), None)
                    if d: break
                case "external":
                    d = next((d for d in DISPLAYS if d.type == DisplayType.EXTERNAL), None)
                    if d: break
                case _:
                    prefix = "model:"
                    if s.startswith(prefix):
                        d = next((d for d in DISPLAYS if d.name == s[len(prefix):]), None)
                        if d: break
                    else:
                        raise FatalErr(f"misconfigured brightness sync strategy [{s}]")

        if d is not None:
            target = d.get_brightness(offset_normalized=True)
        elif target == -1:
            LOGGER.info(
                f"cannot sync brightnesses as no displays detected for sync strategy [{strat}]"
            )
            return

    LOGGER.debug(f"syncing brightnesses at {target}%")
    await TASK_QUEUE.put(["set", target])


async def init_displays_sim(sim) -> None:
    global DISPLAYS

    ndisplays: int = sim.ndisplays

    LOGGER.debug(f"initing {ndisplays} simulated displays...")
    displays: List[SimulatedDisplay] = [
        SimulatedDisplay(f"sim-{i}", CONF) for i in range(ndisplays)
    ]

    futures: List[Task[None]] = [
        asyncio.create_task(d.init(sim.initial_brightness)) for d in displays
    ]
    await wait_and_reraise(futures)

    DISPLAYS = displays
    LOGGER.debug(
        f"...initialized {len(displays)} simulated display{'' if len(displays) == 1 else 's'}"
    )


async def resolve_single_internal_display_raw() -> RawDisplay:
    d = await get_raw_displays()
    return _filter_internal_display(d, BackendType.RAW)

    #  alternative logic by verifying internal display via device path, not device name:
    # device_dirs = glob.glob(CONF.raw_device_dir + '/*')
    # displays = []
    # for i in device_dirs:
        # name = os.path.basename(i)
        # i = Path(i).resolve()
        # if (i.exists() and  # potential dead symlink
                # next((True for segment in i.parts if "eDP-" in segment), False)):  # verify is internal/laptop display, alternative detection
            # displays.append(RawDisplay(name, CONF))  # or RawDisplay(i.name, CONF)

    # assert len(displays) == 1, f'found {len(displays)} raw backlight devices, expected 1'
    # return displays[0]


def _filter_by_backend_type(
    displays: List[TDisplay], bt: BackendType
) -> List[TDisplay]:
    return list(filter(lambda d: d.backend == bt, displays))


def _filter_by_display_type(
    displays: List[TDisplay], dt: DisplayType
) -> List[TDisplay]:
    return list(filter(lambda d: d.type == dt, displays))


def _filter_internal_display(
    disp: List[TNonDDCDisplay], provider: BackendType
) -> TNonDDCDisplay:
    displays: List[TNonDDCDisplay] = _filter_by_display_type(disp, DisplayType.INTERNAL)
    assert len(displays) == 1, (
        f"found {len(displays)} laptop/internal displays w/ {provider}, expected 1"
    )
    return displays[0]


async def resolve_single_internal_display_brillo() -> BrilloDisplay:
    d = await get_brillo_displays()
    return _filter_internal_display(d, BackendType.BRILLO)


async def resolve_single_internal_display_bctl() -> BCTLDisplay:
    d = await get_bctl_displays()
    return _filter_internal_display(d, BackendType.BRIGHTNESSCTL)


async def get_raw_displays() -> List[RawDisplay]:
    device_dirs: List[str] = glob.glob(CONF.raw_device_dir + "/*")
    assert len(device_dirs) > 0, "no backlight-capable raw devices found"

    return [
        RawDisplay(d, CONF) for d in device_dirs if await aios.path.exists(d)
    ]  # exists() check to deal with dead symlinks


async def get_brillo_displays() -> List[BrilloDisplay]:
    out, err, code = await run_cmd(["brillo", "-Ll"], throw_on_err=True, logger=LOGGER)
    out = out.splitlines()
    assert len(out) > 0, "no backlight-capable devices found w/ brillo"

    return [
        BrilloDisplay(os.path.basename(i), CONF)
        for i in out
        if await aios.path.exists(Path(CONF.raw_device_dir, i))
    ]  # exists() check to deal with dead symlinks


async def get_bctl_displays() -> List[BCTLDisplay]:
    cmd = ["brightnessctl", "--list", "--machine-readable", "--class=backlight"]
    out, err, code = await run_cmd(cmd, throw_on_err=True, logger=LOGGER)
    out = out.splitlines()
    assert len(out) > 0, "no backlight-capable devices found w/ brightnessctl"

    return [
        BCTLDisplay(i, CONF)
        for i in out
        if await aios.path.exists(Path(CONF.raw_device_dir, i.split(",")[0]))
    ]  # exists() check to deal with dead symlinks


async def get_ddcutil_displays(ignore_internal: bool) -> List[Display]:
    bus_path_prefix = CONF.ddcutil_bus_path_prefix
    displays: List[Display] = []
    in_invalid_block = False
    d: DDCDisplay | None = None
    out, err, code = await run_cmd(
        ["ddcutil", "--brief", "detect"], throw_on_err=False, logger=LOGGER
    )
    if code != 0:
        if err and "ddcutil requires module i2c" in err:
            raise FatalErr("ddcutil requires i2c-dev kernel module to be loaded")
        LOGGER.error(err)
        raise CmdErr(
            f"ddcutil failed to list/detect devices (exit code {code})", code, err
        )

    for line in out.splitlines():
        if d:
            if line.startswith("   I2C bus:"):
                i = line.find(bus_path_prefix)
                d.bus = line[len(bus_path_prefix) + i :]
            elif line.startswith("   Monitor:"):
                d.name = line.split()[1]
                displays.append(d)
                d = None  # reset
            elif not line:  # block end
                raise FatalErr(
                    f"could not finalize display [{d.id}] - [ddcutil --brief] output has likely changed"
                )
        elif in_invalid_block:  # try to detect laptop internal display
            if not line:
                in_invalid_block = False
            # note matching against "eDP" in "DRM connector" line is not infallible, see https://github.com/rockowitz/ddcutil/issues/547#issuecomment-3253325547
            # expected line will be something like "   DRM connector:    card0-eDP-1"
            elif re.fullmatch(
                r"\s+DRM connector:\s+[a-z0-9]+-eDP-\d+", line
            ):  # i.e. "is this a laptop display?"
                match CONF.internal_display_ctl:
                    case InternalDisplayCtl.RAW:
                        displays.append(await resolve_single_internal_display_raw())
                    case InternalDisplayCtl.BRIGHTNESSCTL:
                        displays.append(await resolve_single_internal_display_bctl())
                    case InternalDisplayCtl.BRILLO:
                        displays.append(await resolve_single_internal_display_brillo())
                in_invalid_block = False
        elif line.startswith("Display "):
            d = DDCDisplay(line.strip(), CONF)
        elif line == "Invalid display" and not ignore_internal:
            # start processing one of the 'Invalid display' blocks:
            in_invalid_block = True
    if d:  # sanity
        raise FatalErr(f"display [{d.id}] defined but not finalized")
    return displays


async def display_op[T](
    op: Callable[[Display], Coroutine[Any, Any, T]],
    disp_filter: Callable[[Display], bool] = lambda _: True,
) -> tuple[List[Task[T]], List[Display]]:
    displays = list(filter(disp_filter, DISPLAYS))
    if not displays:
        raise PayloadErr(
            "no displays for given filter found",
            [1, "no displays for given filter found"],
        )
    futures: List[Task[T]] = [asyncio.create_task(op(d)) for d in displays]
    await wait_and_reraise(futures)
    return futures, displays


def get_disp_filter(opts: Opts | int) -> Callable[[Display], bool]:
    return lambda d: not (
        (opts & Opts.IGNORE_INTERNAL and d.type == DisplayType.INTERNAL)
        or opts & Opts.IGNORE_EXTERNAL and d.type == DisplayType.EXTERNAL
    )


async def execute_tasks(tasks: List[list]) -> None:
    delta: int = 0
    target: int | None = None
    init_retry: None | RetryAsync = None
    sync: bool = False
    opts = 0
    for t in tasks:
        match t:
            case ["delta", opts, d]:  # change brightness by delta %
                delta += d
            case ["delta", d]:  # change brightness by delta %
                delta += d
            case ["up", v]:  # None | int>0
                v = v if v is not None else CONF.brightness_step
                delta += v
            case ["down", v]:  # None | int>0
                v = v if v is not None else CONF.brightness_step
                delta -= v
            case ["set", opts, target]:  # set brightness to a % value
                delta = 0  # cancel all previous deltas
            case ["set", target]:  # set brightness to a % value
                delta = 0  # cancel all previous deltas
            # case ['setmon', display_id, value]:
                # d = next((d for d in DISPLAYS if d.id == display_id), None)
                # if d:
                    # futures.append(asyncio.create_task(d.set_brightness(value)))
            # TODO: perhaps it's safer to remove on_exhaustion from init calls and allow the daemon to crash?:
            case ["init"]:  # re-init displays
                init_retry = get_retry(0, 0, on_exhaustion=True)
            case ["init", retry, sleep]:  # re-init displays
                init_retry = get_retry(retry, sleep, on_exhaustion=True)
            case ["sync"]:
                sync = True
            case ["kill"]:
                sys.exit(0)
            case _:
                LOGGER.error(f"unexpected task {t}")

    if init_retry and isinstance(await init_retry(init_displays), Exception):
        return

    if sync:
        await sync_displays()

    if target is not None:
        target += delta
        # futures = [asyncio.create_task(d.set_brightness(target)) for d in DISPLAYS]
        r = RetryAsync(
            RetriableException,
            retries=1,
            on_exception=(init_displays, OnErrOpts.RUN_ON_LAST_TRY),
        )  # note setting absolute value is retriable
        f = lambda d: d.set_brightness(target)
    elif delta != 0:
        # futures = [asyncio.create_task(d.adjust_brightness(delta)) for d in DISPLAYS]
        r = RetryAsync(
            RetriableException,
            retries=0,
            on_exception=(init_displays, OnErrOpts.RUN_ON_LAST_TRY),
        )
        f = lambda d: d.adjust_brightness(delta)
    else:
        return

    # retry path: {
    number_tasks = f"{len(tasks)} task{'' if len(tasks) == 1 else 's'}"
    LOGGER.debug(f"about to execute() {number_tasks}...")
    try:
        futures, _ = await r(display_op, f, get_disp_filter(opts))
        LOGGER.debug(f"...executed {number_tasks}")
    except Exception as e:
        LOGGER.error(f"...error executing tasks: {e}")
        return

    # } non-retry path: {
    # if not futures:
        # await TASK_QUEUE.put(["init"])
        # return
    # number_tasks = f'{len(tasks)} task{"" if len(tasks) == 1 else "s"}'
    # LOGGER.debug(f'about to execute() {number_tasks}...')
    # try:
        # await wait_and_reraise(futures)
        # LOGGER.debug(f"...executed {number_tasks}")
    # except Exception as e:
        # LOGGER.error(f"...error executing tasks: {e}")
        # await TASK_QUEUE.put(["init"])
        # return
    #}

    brightnesses: List[int] = sorted([f.result() for f in futures])
    if not opts & Opts.NO_TRACK and (brightnesses[-1] - brightnesses[0]) <= 2:
        print(f'yooooooooooooo: {brightnesses}')
        CONF.state.last_set_brightness = brightnesses[0]
    if not opts & Opts.NO_NOTIFY:
        await NOTIF.notify_change(brightnesses[0])  # TODO: shouldn't we consolidate the value?
    if CONF.sync_brightness:
        await sync_displays()


async def process_q() -> NoReturn:
    consumption_window: float = CONF.msg_consumption_window_sec
    while True:
        tasks: List[list] = []
        t: list = await TASK_QUEUE.get()
        tasks.append(t)
        while True:
            try:
                t = await asyncio.wait_for(TASK_QUEUE.get(), consumption_window)
                tasks.append(t)
            except TimeoutError:
                break
        async with LOCK:
            await execute_tasks(tasks)


def get_retry(
    retries, sleep, on_exhaustion: bool | Callable = False, on_exception=None
) -> RetryAsync:
    return RetryAsync(
        RetriableException,
        retries=retries,
        backoff=sleep,
        on_exhaustion=on_exhaustion,
        on_exception=on_exception,
    ).__call__


def handle_failure_after_retries(e: Exception):
    if isinstance(e, PayloadErr):
        return e.payload
    return [1, str(e)]


async def process_client_commands(err_event: Event) -> None:
    init_displays_retry_handler = get_retry(2, 0.3, True)

    # this wrapper is so exceptions from serve_forever() callbacks (which are not
    # awaited on) get propagated up to our taskgroup.
    async def wrapped_client_connected_cb(*args):
        try:
            return await process_client_command(*args)
        except Exception as error:
            err_event.err = error
            err_event.set()
            raise

    async def process_client_command(reader, writer):
        async def _send_response(payload: list):
            response = json.dumps(payload, separators=(",", ":"))
            LOGGER.debug(f"responding: {response}")
            writer.write(response.encode())
            await writer.drain()
            writer.write_eof()
            writer.close()
            await writer.wait_closed()

        async def disp_op[T](
            op: Callable[[Display], Coroutine[Any, Any, T]],
            disp_filter: Callable[[Display], bool],
            payload_creator: Callable[
                [List[Task[T]], List[Display]], List[int | str]
            ] = lambda *_: [0],
        ):
            futures, displays = await display_op(op, disp_filter)
            return payload_creator(futures, displays)

        data = (await reader.read()).decode()
        if not data:
            return
        data = json.loads(data)
        LOGGER.debug(f"received task {data} from client")
        init_displays_retry = partial(init_displays_retry_handler, init_displays)
        payload = [1]
        match data:
            case ["get", opts]:
                async with LOCK:
                    with suppress(RetriableException):
                        payload = [0, *get_brightness(opts)]
            case ["setvcp", retry, sleep, *params]:
                r = get_retry(
                    retry, sleep, handle_failure_after_retries, init_displays_retry
                )
                async with LOCK:
                    payload = await r(
                        disp_op,
                        lambda d: d._set_vcp_feature(*params),
                        lambda d: d.backend == BackendType.DDCUTIL,
                    )

            case ["getvcp", retry, sleep, *params]:
                r = get_retry(
                    retry, sleep, handle_failure_after_retries, init_displays_retry
                )

                async with LOCK:
                    payload = await r(
                        disp_op,
                        lambda d: d._get_vcp_feature(*params),
                        lambda d: d.backend == BackendType.DDCUTIL,
                        lambda futures, displays: [
                            0,
                            *[
                                f"{displays[i].id},{j.result()}"
                                for i, j in enumerate(futures)
                            ],
                        ],
                    )
            case ["set_for", retry, sleep, disp_to_brightness]:
                r = get_retry(
                    retry, sleep, handle_failure_after_retries, init_displays_retry
                )
                async with LOCK:
                    payload = await r(
                        disp_op,
                        lambda d: d.set_brightness(disp_to_brightness[d.id]),
                        lambda d: d.id in disp_to_brightness,
                    )
            case ["set_for_async", disp_to_brightness]:
                # TODO: consider passing OnErrOpts.RUN_ON_LAST_TRY to retry opts; with that we might even change to retries=0
                r = get_retry(1, 0.5, True, init_displays_retry)
                async with LOCK:
                    await r(
                        display_op,
                        lambda d: d.set_brightness(disp_to_brightness[d.id]),
                        lambda d: d.id in disp_to_brightness,
                    )
                    return
            case _:
                LOGGER.debug("placing task in queue...")
                await TASK_QUEUE.put(data)
                return
        await _send_response(payload)

    server = await asyncio.start_unix_server(wrapped_client_connected_cb, SOCKET_PATH)
    await server.serve_forever()


async def delta_brightness(delta: int):
    LOGGER.debug(
        f"placing brightness change in queue for delta {'+' if delta > 0 else ''}{delta}"
    )
    await TASK_QUEUE.put(["delta", delta])


async def terminate():
    LOGGER.info("placing termination request in queue")
    await TASK_QUEUE.put(["kill"])
    # alternatively, ignore existing queue and terminate immediately:
    # try:
        # await write_state(CONF, DISPLAYS)
    # finally:
        # os._exit(0)


# note raw values only make sense when asked for individual displays, as
# we can't really collate them into a single value as the scales potentially differ
def get_brightness(opts) -> List[int | List[str | int]]:
    displays = list(filter(get_disp_filter(opts), DISPLAYS))

    if not displays:
        return []
    elif opts & Opts.GET_INDIVIDUAL:
        # return [f'{d.id},{d.get_brightness(raw=opts & Opts.GET_RAW, offset_normalized=opts & Opts.GET_OFFSET_NORMALIZED)}' for d in displays]
        return [
            [
                d.id,
                d.get_brightness(
                    raw=opts & Opts.GET_RAW,
                    offset_normalized=opts & Opts.GET_OFFSET_NORMALIZED,
                ),
            ]
            for d in displays
        ]

    # either ignore last_set_brightness... {
    values: List[int] = [
        d.get_brightness(offset_normalized=opts & Opts.GET_OFFSET_NORMALIZED)
        for d in displays
    ]
    if same_values(values):
        val = values[0]
    else:
    # } ...or use it: {
    # val: int = CONF.state.last_set_brightness
    # if val == -1:  # i.e. we haven't explicitly set it to anything yet
        # values: List[int] = [d.get_brightness(offset_normalized=opts & Opts.GET_OFFSET_NORMALIZED) for d in displays]
    #}
        match CONF.get_strategy:
            case GetStrategy.MEAN:
                val = int(fmean(values))
            case GetStrategy.LOW:
                val = min(values)
            case GetStrategy.HIGH:
                val = max(values)

    return [val]


async def periodic_init(period: int) -> NoReturn:
    delta_threshold_sec = period - 1 - CONF.msg_consumption_window_sec
    if delta_threshold_sec <= 0:
        raise FatalErr("[periodic_init_sec] value too low")

    while True:
        await asyncio.sleep(period)
        if unix_time_now() - LAST_INIT_TIME >= delta_threshold_sec:
            LOGGER.debug("placing periodic [init] task on the queue...")
            await TASK_QUEUE.put(["init", 1, 0.5])


async def catch_err(err_event: Event) -> None:
    await err_event.wait()
    raise err_event.err


async def run() -> None:
    try:
        validate_ext_deps()
        init_displays_retry_handler = get_retry(5, 0.8)
        await init_displays_retry_handler(init_displays)
        err_event: Event = asyncio.Event()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(process_q())
            tg.create_task(catch_err(err_event))
            tg.create_task(process_client_commands(err_event))
            if CONF.monitor_udev:
                debounced = Debouncer(delay=CONF.udev_event_debounce_sec)
                f = partial(debounced, TASK_QUEUE.put, ["init", 2, 0.5])
                tg.create_task(monitor_udev_events("drm", "change", f))
            if CONF.periodic_init_sec:
                tg.create_task(periodic_init(CONF.periodic_init_sec))

            loop: AbstractEventLoop = asyncio.get_running_loop()
            loop.add_signal_handler(
                signal.SIGUSR1,
                lambda: tg.create_task(delta_brightness(CONF.brightness_step)),
            )
            loop.add_signal_handler(
                signal.SIGUSR2,
                lambda: tg.create_task(delta_brightness(-CONF.brightness_step)),
            )
            loop.add_signal_handler(signal.SIGINT, lambda: tg.create_task(terminate()))
            loop.add_signal_handler(signal.SIGTERM, lambda: tg.create_task(terminate()))
    except* (ExitableErr, FatalErr) as exc_group:
        LOGGER.debug(f"{len(exc_group.exceptions)} errs caught in exc group")

        ee = exc_group.exceptions[0]
        if isinstance(ee, ExitableErr):
            exit_code: int = ee.exit_code
        else:
            exit_code: int = CONF.fatal_exit_code
        LOGGER.debug(f"{type(ee).__name__} caught, exiting with code {exit_code}...")
        await NOTIF.notify_err(ee)
        sys.exit(exit_code)
    except* SystemExit:
        raise
    except* Exception as exc_group:
        LOGGER.error("caught following unhandled errors in exc_group:")
        for i, e in enumerate(exc_group.exceptions):
            LOGGER.error(f"{i}.: {e}")
            await NOTIF.notify_err(e)
        sys.exit(1)
    finally:
        await write_state(CONF, DISPLAYS)


# top-level err handler that's caught for stuff ran prior to task group.
# note unhandled exceptions in run() also get propageted up here
def root_exception_handler(
    type_: type[BaseException], value: BaseException, tbt: TracebackType | None
) -> None:
    # LOGGER.debug('root exception handler triggered')
    if isinstance(value, ExitableErr):
        traceback.print_tb(tbt)
        # NOTIF.notify_err_sync(value)
        sys.exit(value.exit_code)
    sys.__excepthook__(type_, value, tbt)


def main(debug=False, sim_conf: SimConf | None = None) -> None:
    global SINGLETON_LOCK
    global TASK_QUEUE
    global LOCK
    global CONF
    global NOTIF

    # sys.excepthook = root_exception_handler

    SINGLETON_LOCK = singleton.SingleInstance()
    TASK_QUEUE = asyncio.Queue()
    LOCK = Lock()

    CONF = load_config(load_state=True)
    CONF.sim = sim_conf
    NOTIF = Notif(CONF.notify)

    log_lvl = logging.DEBUG if debug else getattr(logging, CONF.log_lvl)
    logging.basicConfig(stream=sys.stdout, level=log_lvl)

    asyncio.run(run())
