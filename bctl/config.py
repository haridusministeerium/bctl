import os
import glob
import json
import itertools
import logging
from enum import StrEnum, auto
from collections.abc import Iterable
import aiofiles as aiof
from pydantic import BaseModel
from pydash import py_
from datetime import datetime
from logging import Logger
from bctl.common import CACHE_PATH
from bctl.exceptions import FatalErr

LOGGER: Logger = logging.getLogger(__name__)

STATE_VER = 1  # bump this whenever persisted state data structure changes


class SimConf(BaseModel):
    ndisplays: int
    wait_sec: float
    initial_brightness: int
    failmode: str | None
    exit_code: int


class State(BaseModel):
    timestamp: int = 0
    ver: int = STATE_VER
    last_set_brightness: int = -1  # value we've set all displays' brightnesses (roughly) to;
                                   # -1 if brightnesses differ or we haven't set brightness using bctl yet
    offsets: dict[str, list[int]] = {}  # display.id->[offset, eoffset] mappings;
                                        # note storing & rehydrating eoffsets is only relevant
                                        # when sync_brightness=False and first change upon
                                        # service restart is delta(), not set(); as syncing
                                        # already invokes set(), then that'll apply the correct
                                        # eoffset


class NotifyIconConf(BaseModel):
    error: str = "gtk-dialog-error"
    root_dir: str = ""
    brightness_full: str = "notification-display-brightness-full.svg"
    brightness_high: str = "notification-display-brightness-high.svg"
    brightness_medium: str = "notification-display-brightness-medium.svg"
    brightness_low: str = "notification-display-brightness-low.svg"
    brightness_off: str = "notification-display-brightness-off.svg"


class NotifyConf(BaseModel):
    enabled: bool = True
    on_fatal_err: bool = True  # whether desktop notifications should be shown on fatal errors
    timeout_ms: int = 4000
    icon: NotifyIconConf = NotifyIconConf()


class OffsetType(StrEnum):
    SOFT = auto()  # allows offset screen to go over its boundary, e.g. brightness of
                   # a screen w/ offset=10 can still be lowered all the way to 0%
    HARD = auto()  # offset sets a hard limit; e.g. offset=20 means this screen's
                   # brightness is not allowed to go below 20%


class OffsetConf(BaseModel):
    type: OffsetType = OffsetType.HARD
    offsets: dict[str, int] = {}  # criteria -> offset rules; accepted keys are "internal", "external", "any", "id:<id>";
                                  # <id> being [ddcutil --brief detect] cmd "Monitor:" value (or alias).
                                  # offset of 20 means those displays' brightness will be 20% over the set limit;
                                  # likewise -20 means brightness will be 20% lower than the set limit.

    enabled_if: str = ""  # global rule to disable all offsets if this expression does not evaluate true;
                          # will be eval()'d in init_displays(), dangerous!
                          # e.g.:
                          # - "__import__('socket').gethostname() in ['hostname1', 'hostname2']"
                          # - "os.uname().nodename in ['hostname1', 'hostname2']"
                          # - "displays"
    disabled_if: str = ""  # global rule to disable all offsets if this expression evaluates true;
                           # will be eval()'d in init_displays(), dangerous!
                           # e.g.:
                           # - "len(displays) <= 1"


class MainDisplayCtl(StrEnum):
    DDCUTIL = auto()
    RAW = auto()
    BRIGHTNESSCTL = auto()
    BRILLO = auto()


class InternalDisplayCtl(StrEnum):
    RAW = auto()
    BRIGHTNESSCTL = auto()
    BRILLO = auto()


class GetStrategy(StrEnum):
    MEAN = auto()  # return arithmetic mean
    LOW = auto()   # return lowest
    HIGH = auto()  # return highest


class Conf(BaseModel):
    log_lvl: str = "INFO"  # daemon log level, doesn't apply to the client
    ddcutil_bus_path_prefix: str = "/dev/i2c-"  # prefix to the bus number
    ddcutil_brightness_feature: str = "10"
    ddcutil_svcp_flags: list[str] = [  # flags passed to [ddcutil setvcp] commands
        "--skip-ddc-checks"
    ]
    ddcutil_gvcp_flags: list[str] = []  # flags passed to [ddcutil getvcp] commands
                                        # you might also want to consider ddcutil configuration; see https://www.ddcutil.com/config_file/
    monitor_udev: bool = True  # monitor udev events for drm subsystem to detect ext. display (dis)connections
    udev_event_debounce_sec: float = 3.0  # both for debouncing & delay; have experienced missed ext. display detection w/ 1.0, but it's flimsy regardless
    periodic_init_sec: int = 0  # periodically re-init/re-detect monitors; 0 to disable
    sync_brightness: bool = False  # keep all displays' brightnesses at same value/synchronized
    sync_strategy: tuple[str, ...] = ("mean",)  # if displays' brightnesses differ and are synced, what value to sync them to; only active if sync_brightness=True;
                                # first matched strategy is applied, i.e. can define as ["id:AUS:PA278QV:L9GMQA215221", "internal", "mean"]
                                # - mean = set to arithmetic mean
                                # - low = set to lowest
                                # - high = set to highest
                                # - internal = set to the internal screen value
                                # - external = set to _a_ external screen value
                                # - id:<id> = set to <id> screen value; <id> being [ddcutil --brief detect] cmd "Monitor:" value (or alias)
    get_strategy: GetStrategy = GetStrategy.MEAN  # if displays' brightnesses differ and are queried via get command,
                                                  # what single value to return to represent current brightness level;
    notify: NotifyConf = NotifyConf()
    offset: OffsetConf = OffsetConf()
    msg_consumption_window_sec: float = 0.1  # can be set to 0 if no delay/window is required
    brightness_step: int = 5  # %
    ignored_displays: set[str] = set()  # either [ddcutil --brief detect] cmd "Monitor:" value, or <device> in /sys/class/backlight/<device>
    ignore_internal_display: bool = False  # do not control internal (i.e. laptop) display if available
    ignore_external_display: bool = False  # do not control external display(s) if available
    aliases: dict[str, tuple[str, ...]] = {}  # aliases to display IDs; cannot use "laptop" or "internal",
                                              # as those are automatically assigned as aliases to laptop display
    main_display_ctl: MainDisplayCtl = MainDisplayCtl.DDCUTIL
    internal_display_ctl: InternalDisplayCtl = InternalDisplayCtl.RAW  # only used if main_display_ctl=ddcutil and we're a laptop
    raw_device_dir: str = "/sys/class/backlight"  # used if main_display_ctl=raw OR
                                                  # (main_display_ctl=ddcutil AND internal_display_ctl=raw AND we're a laptop)
    fatal_exit_code: int = 100  # exit code signifying fatal exit that should not be retried/restarted;
                                # you might want to use this value in systemd unit file w/ RestartPreventExitStatus config
    sim: SimConf | None = None  # simulation config, will be set by sim client
    state_f_path: str = f"{CACHE_PATH}/bctld.state"  # state that should survive restarts are stored here
    state: State = State()  # do not set, will be read in from state_f_path
    max_state_age_sec: int = 60  # only use persisted state if it's younger than this; <= 0 to always use state regardless of its age


def load_config(load_state: bool = False) -> Conf:
    configs = [
        _read_dict_from_file(c) for c in sorted(glob.glob(_conf_path() + "/*.json"))
    ]
    conf = py_.merge({}, *configs)
    conf = Conf.model_validate(conf)

    if load_state:
        conf.state = _load_state(conf)
    flat_aliases = list(itertools.chain.from_iterable(conf.aliases.values()))
    if find_dupes(flat_aliases):
        raise FatalErr("[aliases] definitions contain duplicates")
    reserved_aliases = ["internal", "laptop"]
    if bool(set(reserved_aliases) & set(flat_aliases)):
        raise FatalErr(f"[aliases] config cannot contain reserved values {reserved_aliases}")

    # LOGGER.error(f"effective config: {conf}")
    return conf


def _conf_path() -> str:
    xdg_dir = os.environ.get("XDG_CONFIG_HOME", f"{os.environ['HOME']}/.config")
    return xdg_dir + "/bctl"


def _load_state(conf: Conf) -> State:
    try:
        s = State.model_validate_json(_read_json_bytes_from_file(conf.state_f_path))

        t = s.timestamp
        v = s.ver
        is_state_young_enough: bool = (unix_time_now() - t <= conf.max_state_age_sec) if conf.max_state_age_sec > 0 else True
        if is_state_young_enough and v == STATE_VER:
            LOGGER.debug(f"hydrated state from disk: {s}")
            return s
        raise
    except Exception:
        return State()


def find_dupes[T](i: Iterable[T]) -> list[T]:
    j = sorted(i)
    return list(set(j[::2]) & set(j[1::2]))


# note this one doesn't flatten the input
def intersect[T](iterable_of_iterables: Iterable[Iterable[T]]) -> set[T]:
    iterator = iter(iterable_of_iterables)
    return set(next(iterator)).intersection(*iterator)


async def write_state(conf: Conf) -> None:
    conf.state.timestamp = unix_time_now()

    try:
        LOGGER.debug("storing state...")
        payload = conf.state.model_dump_json(indent=2)
        async with aiof.open(conf.state_f_path, mode="w") as f:
            await f.write(payload)
        LOGGER.debug("...state stored")
    except IOError:
        raise


def _read_json_bytes_from_file(file_loc: str) -> bytes:
    if not (os.path.isfile(file_loc) and os.access(file_loc, os.R_OK)):
        return b"{}"

    try:
        with open(file_loc, "rb") as f:
            return f.read()
    except Exception:
        LOGGER.error(f"error trying to read json as bytes from {file_loc}")
        return b"{}"


def _read_dict_from_file(file_loc: str) -> dict:
    if not (os.path.isfile(file_loc) and os.access(file_loc, os.R_OK)):
        return {}

    try:
        with open(file_loc, "r") as f:
            return json.load(f)
    except Exception:
        LOGGER.error(f"error trying to parse json from {file_loc}")
        return {}


def unix_time_now() -> int:
    return int(datetime.now().timestamp())
