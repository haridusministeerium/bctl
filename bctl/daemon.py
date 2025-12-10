import asyncio
import traceback
import signal
import sys
import json
import logging
from functools import partial
from retry_deco import RetryAsync, OnErrOpts
from logging import Logger
from types import TracebackType
from asyncio import AbstractEventLoop, Task, Queue, Lock, Event
from typing import NoReturn, Callable, Coroutine, Any
from collections.abc import Sequence
from tendo import singleton
from statistics import fmean
from contextlib import suppress
from bctl.init import resolve_displays
from bctl.debouncer import Debouncer
from bctl.udev_monitor import monitor_udev_events
from bctl.display import BackendType, DisplayType, Display, get_disp_filter
from bctl.common import (
    same_values,
    assert_cmd_exist,
    wait_and_reraise,
    Opts,
    SOCKET_PATH,
)
from bctl.config import (
    Conf,
    SimConf,
    load_config,
    write_state,
    unix_time_now,
    GetStrategy,
)
from bctl.exceptions import (
    ExitableErr,
    FatalErr,
    PayloadErr,
    RetriableException,
)
from bctl.notify import Notif

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
    DISPLAYS = await resolve_displays(CONF)

    if CONF.sync_brightness:
        await sync_displays()
    # TODO: is resetting last_set_brightness reasonable here? even when a new display
    #       gets connected, doesn't it still remain our last requested brightness target?
    #       suppose only place where reset is required is if our get_brightness()
    #       returns the last_set_brightness if it's set, which in this case would
    #       become invalid. so it really depends on how last_set_brightness is used.
    elif DISPLAYS and not same_values([d.get_brightness() for d in DISPLAYS]):
        CONF.state.last_set_brightness = -1  # reset, as potentially newly added display could have a different value

    LAST_INIT_TIME = unix_time_now()


async def sync_displays(opts=0) -> None:
    displays = list(filter(get_disp_filter(opts), DISPLAYS)) if opts else DISPLAYS
    if len(displays) <= 1:
        return
    values: list[int] = sorted(d.get_brightness() for d in displays)
    #if same_values(values):
    if values[-1] - values[0] <= 1:  # allow for some flex
        return

    target: int = CONF.state.last_set_brightness
    if target == -1:  # i.e. we haven't explicitly set it to anything yet
        d: Display | None = None
        for strat in CONF.sync_strategy:
            match strat:
                case "mean":
                    target = round(fmean(values))
                    break
                case "low":
                    target = min(values)
                    break
                case "high":
                    target = max(values)
                    break
                case "internal":
                    d = next((d for d in displays if d.type is DisplayType.INTERNAL), None)
                    if d: break
                case "external":
                    d = next((d for d in displays if d.type is DisplayType.EXTERNAL), None)
                    if d: break
                case _:
                    prefix = "id:"
                    if strat.startswith(prefix):
                        d = next((d for d in displays if strat[len(prefix):] in d.names), None)
                        if d: break
                    else:
                        raise FatalErr(
                            f"misconfigured brightness sync strategy [{strat}]"
                        )

        if d is not None:
            target = d.get_brightness()
        elif target == -1:
            LOGGER.info(
                f"cannot sync brightnesses as no displays detected for sync strategy [{CONF.sync_strategy}]"
            )
            return
        LOGGER.debug(f"syncing using [{strat}] strategy at {target}%...")
    else:
        LOGGER.debug(f"syncing using last-set-value at {target}%")

    await TASK_QUEUE.put(["set", opts, target])


async def display_op[T](
    op: Callable[[Display], Coroutine[Any, Any, T]],
    disp_filter: Callable[[Display], bool] = lambda _: True,
) -> tuple[list[Task[T]], list[Display]]:
    displays = list(filter(disp_filter, DISPLAYS))
    if not displays:
        raise PayloadErr(
            "no displays for given filter found",
            [1, "no displays for given filter found"],
        )
    futures: list[Task[T]] = [asyncio.create_task(op(d)) for d in displays]
    await wait_and_reraise(futures)
    return futures, displays


async def execute_tasks(tasks: list[list]) -> None:
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
                # d = next((d for d in DISPLAYS if display_id in d.names), None)
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

    await post_set(futures, opts)


async def post_set(futures: Task[int], opts) -> None:
    brightnesses: list[int] = sorted(f.result() for f in futures)
    if not opts & Opts.NO_TRACK and (brightnesses[-1] - brightnesses[0]) <= 1:
        CONF.state.last_set_brightness = brightnesses[0]
    if not opts & Opts.NO_NOTIFY:
        await NOTIF.notify_change(brightnesses[0])  # TODO: shouldn't we consolidate the value?
    if CONF.sync_brightness and not opts & Opts.NO_SYNC:
        await sync_displays(opts)


async def process_q() -> NoReturn:
    consumption_window: float = CONF.msg_consumption_window_sec
    while True:
        tasks: list[list] = []
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
    init_displays_retry = partial(init_displays_retry_handler, init_displays)

    # this wrapper is so exceptions from serve_forever() callbacks (which are not
    # awaited on) get propagated up to our taskgroup.
    async def wrapped_client_connected_cb(*args):
        try:
            return await process_client_command(*args)
        except ConnectionResetError as error:
            pass  # e.g. if client process is terminated during communication
        except Exception as error:
            err_event.err = error
            err_event.set()
            raise

    async def disp_op[T](
        op: Callable[[Display], Coroutine[Any, Any, T]],
        disp_filter: Callable[[Display], bool],
        payload_creator: Callable[
            [list[Task[T]], list[Display]], list[int | str]
        ] = lambda *_: [0],
    ):
        futures, displays = await display_op(op, disp_filter)
        return payload_creator(futures, displays)

    async def process_client_command(reader, writer) -> None:
        async def _close_socket() -> None:
            # self.logger.debug("closing the connection")
            writer.write_eof()
            writer.close()
            await writer.wait_closed()

        async def _send_response(payload: list) -> None:
            response = json.dumps(payload, separators=(",", ":"))
            LOGGER.debug(f"responding: {response}")
            writer.write(response.encode())
            await writer.drain()
            await _close_socket()

        data = (await reader.read()).decode()
        if not data:
            return
        data = json.loads(data)
        LOGGER.debug(f"received task {data} from client")
        payload = [1]
        match data:
            case ["get"]:
                async with LOCK:
                    with suppress(RetriableException):
                        payload = [0, *get_brightness(0, False)]
            case ["get", opts, *displays]:
                async with LOCK:
                    with suppress(RetriableException):
                        payload = [0, *get_brightness(opts, *displays)]
            case ["setvcp", retry, sleep, displays, *params]:
                r = get_retry(
                    retry, sleep, handle_failure_after_retries, init_displays_retry
                )
                async with LOCK:
                    payload = await r(
                        disp_op,
                        lambda d: d._set_vcp_feature(*params),
                        lambda d: d.backend is BackendType.DDCUTIL
                        and (any(x in d.names for x in displays) if displays else True),
                    )
            case ["getvcp", retry, sleep, displays, *params]:
                r = get_retry(
                    retry, sleep, handle_failure_after_retries, init_displays_retry
                )

                async with LOCK:
                    payload = await r(
                        disp_op,
                        lambda d: d._get_vcp_feature(*params),
                        lambda d: d.backend is BackendType.DDCUTIL
                        and (any(x in d.names for x in displays) if displays else True),
                        lambda futures, displays: [
                            0,
                            *[
                                [displays[i].id, f.result().strip()]
                                for i, f in enumerate(futures)
                            ],
                        ],
                    )
            case ["set_for_sync", retry, sleep, disp_to_brightness]:
                r = get_retry(
                    retry, sleep, handle_failure_after_retries, init_displays_retry
                )
                async with LOCK:
                    payload = await r(
                        disp_op,
                        lambda d: d.set_brightness(
                            next(
                                disp_to_brightness[x]
                                for x in d.names
                                if x in disp_to_brightness
                            )
                        ),
                        lambda d: any(x in disp_to_brightness for x in d.names),
                        lambda futures, displays: [
                            0,
                            *[
                                [displays[i].id, f.result()]
                                for i, f in enumerate(futures)
                            ],
                        ],
                    )
            case ["set_sync", retry, sleep, opts, value]:
                r = get_retry(retry, sleep, on_exception=init_displays_retry)
                async with LOCK:
                    with suppress(RetriableException):
                        futures, displays = await r(
                            display_op,
                            lambda d: d.set_brightness(value),
                        )
                        await post_set(futures, opts)
                        payload = [
                            0,
                            *[
                                [
                                    d.id,
                                    d.get_brightness(
                                        no_offset_normalized=opts
                                        & Opts.GET_NO_OFFSET_NORMALIZED
                                    ),
                                ]
                                for d in displays
                            ],
                        ]
            case ["set_for_async", disp_to_brightness]:
                # TODO: consider passing OnErrOpts.RUN_ON_LAST_TRY to retry opts; with that we might even change to retries=0
                r = get_retry(1, 0.5, True, init_displays_retry)
                async with LOCK:
                    await r(
                        display_op,
                        lambda d: d.set_brightness(
                            next(
                                disp_to_brightness[x]
                                for x in d.names
                                if x in disp_to_brightness
                            )
                        ),
                        lambda d: any(x in disp_to_brightness for x in d.names),
                    )
                await _close_socket()
                return
            case _:
                LOGGER.debug("placing task in queue...")
                await TASK_QUEUE.put(data)
                await _close_socket()
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
        # await write_state(CONF)
    # finally:
        # os._exit(0)


# note raw values only make sense when asked for specific or all displays, as
# we can't really collate them into a single value as the scales potentially differ
def get_brightness(opts, display_names) -> tuple[int] | list[tuple[str, int]]:
    filters: list[Callable[[Display], bool]] = []
    if opts:
        filters.append(get_disp_filter(opts))
    if display_names:
        filters.append(lambda d: any(x in d.names for x in display_names))

    if filters:
        displays = list(filter(lambda d: all(f(d) for f in filters), DISPLAYS))
    else:
        displays = DISPLAYS

    if not displays:
        return []
    elif opts & Opts.GET_ALL or display_names:
        # return [f'{d.id},{d.get_brightness(raw=opts & Opts.GET_RAW, no_offset_normalized=opts & Opts.GET_NO_OFFSET_NORMALIZED)}' for d in displays]
        return [
            (
                d.id,
                d.get_brightness(
                    raw=opts & Opts.GET_RAW,
                    no_offset_normalized=opts & Opts.GET_NO_OFFSET_NORMALIZED,
                ),
            )
            for d in displays
        ]

    # either ignore last_set_brightness... {
    values: list[int] = [
        d.get_brightness(no_offset_normalized=opts & Opts.GET_NO_OFFSET_NORMALIZED)
        for d in displays
    ]
    if same_values(values):
        val = values[0]
    else:
    # } ...or use it: {
    # val: int = CONF.state.last_set_brightness
    # if val == -1:  # i.e. we haven't explicitly set it to anything yet
        # values: list[int] = [d.get_brightness(no_offset_normalized=opts & Opts.GET_NO_OFFSET_NORMALIZED) for d in displays]
    #}
        match CONF.get_strategy:
            case GetStrategy.MEAN:
                val = round(fmean(values))
            case GetStrategy.LOW:
                val = min(values)
            case GetStrategy.HIGH:
                val = max(values)

    return (val,)


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
        await get_retry(5, 0.8)(init_displays)
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
        await write_state(CONF)


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


def main(debug: bool, load_state: bool, sim_conf: SimConf | None = None) -> None:
    global SINGLETON_LOCK
    global TASK_QUEUE
    global LOCK
    global CONF
    global NOTIF

    # sys.excepthook = root_exception_handler

    SINGLETON_LOCK = singleton.SingleInstance()
    TASK_QUEUE = asyncio.Queue()
    LOCK = Lock()

    CONF = load_config(load_state=load_state)
    CONF.sim = sim_conf
    NOTIF = Notif(CONF.notify)

    log_lvl = logging.DEBUG if debug else getattr(logging, CONF.log_lvl)
    logging.basicConfig(stream=sys.stdout, level=log_lvl)

    asyncio.run(run())
