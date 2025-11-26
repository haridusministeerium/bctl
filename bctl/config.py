import os
import json
import logging
import aiofiles as aiof
from pydantic import BaseModel
from datetime import datetime
from logging import Logger
from .common import RUNTIME_PATH

LOGGER: Logger = logging.getLogger(__name__)

STATE_VER = 1  # bump this whenever persisted state data structure changes
TIME_DIFF_DELTA_THRESHOLD_S = 60


class SimConf(BaseModel):
    ndisplays: int
    wait_sec: float
    initial_brightness: int
    failmode: str | None
    exit_code: int


class State(BaseModel):
    timestamp: int = 0
    ver: int = -1
    last_set_brightness: int = -1  # value we've set all displays' brightnesses (roughly) to;
                                   # -1 if brightnesses differ or we haven't set brightness using bctl yet


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


class Conf(BaseModel):
    log_lvl: str = "INFO"  # daemon log level, doesn't apply to the client
    ddcutil_bus_path_prefix: str = "/dev/i2c-"  # prefix to the bus number
    ddcutil_brightness_feature: str = "10"
    ddcutil_svcp_flags: list[str] = [  # flags passed to [ddcutil setvcp] commands
        "--skip-ddc-checks"
    ]
    ddcutil_gvcp_flags: list[str] = []  # flags passed to [ddcutil getvcp] commands
    monitor_udev: bool = True  # monitor udev events for drm subsystem to detect ext. display (dis)connections
    udev_event_debounce_sec: float = 3.0  # both for debouncing & delay; have experienced missed ext. display detection w/ 1.0, but it's flimsy regardless
    periodic_init_sec: int = 0  # periodically re-init/re-detect monitors; 0 to disable
    sync_brightness: bool = False  # keep all displays' brightnesses at same value/synchronized
    sync_strategy: list[str] = ["MEAN"]  # if displays' brightnesses differ and are synced, what value to sync them to; only active if sync_brightness=True;
                                # first matched strategy is used, i.e. can define as ["MODEL:AUS:PA278QV:L9GMQA215221", "INTERNAL", "MEAN"]
                                # - MEAN = set to arithmetic mean
                                # - LOW = set to lowest
                                # - HIGH = set to highest
                                # - INTERNAL = set to the internal screen value
                                # - EXTERNAL = set to _a_ external screen value
                                # - MODEL:<model> = set to <model> screen value; <model> being [ddcutil --brief detect] cmd "Monitor:" value
    get_strategy: str = "MEAN"  # if displays' brightnesses differ and are queried (via get command), what single value to return to represent current brightness level;
                                # 'MEAN' = return arithmetic mean, 'LOW' = return lowest, 'HIGH' = return highest
    notify: NotifyConf = NotifyConf()
    msg_consumption_window_sec: float = 0.1  # can be set to 0 if no delay/window is required
    brightness_step: int = 5  # %
    ignored_displays: list[str] = []  # either [ddcutil --brief detect] cmd "Monitor:" value, or <device> in /sys/class/backlight/<device>
    ignore_internal_display: bool = False  # do not control internal (i.e. laptop) display if available
    ignore_external_display: bool = False  # do not control external display(s) if available
    main_display_ctl: str = "DDCUTIL"  # RAW | DDCUTIL | BRIGHTNESSCTL | BRILLO
    internal_display_ctl: str = "RAW"  # RAW | BRIGHTNESSCTL | BRILLO;  only used if main_display_ctl=DDCUTIL and we're a laptop
    raw_device_dir: str = "/sys/class/backlight"  # used if main_display_ctl=RAW OR
                                                  # (main_display_ctl=DDCUTIL AND internal_display_ctl=RAW AND we're a laptop)
    fatal_exit_code: int = 100  # exit code signifying fatal exit that should not be retried;
                                # you might want to use this value in systemd unit file w/ RestartPreventExitStatus config
    sim: SimConf | None = None  # simulation config, will be set by sim client
    state_f_path: str = f"{RUNTIME_PATH}/bctld.state"  # state that should survive restarts are stored here
    state: State = State()  # do not set, will be read in from state_f_path


def load_config(load_state: bool = False) -> Conf:
    conf = Conf.model_validate_json(_read_json_bytes_from_file(_conf_path()))

    if load_state:
        conf.state = _load_state(conf.state_f_path)

    # LOGGER.debug(f'effective config: {conf}')
    return conf


def _conf_path() -> str:
    xdg_dir = os.environ.get("XDG_CONFIG_HOME", f"{os.environ['HOME']}/.config")
    return xdg_dir + "/bctl/config.json"


def _load_state(file_loc: str) -> State:
    s = State.model_validate_json(_read_json_bytes_from_file(file_loc))

    t = s.timestamp
    v = s.ver
    if unix_time_now() - t <= TIME_DIFF_DELTA_THRESHOLD_S and v == STATE_VER:
        LOGGER.debug(f"hydrated state from disk: {s}")
        return s
    return State()


async def write_state(conf: Conf) -> None:
    data: dict = {
        "timestamp": unix_time_now(),
        "ver": STATE_VER,
        "last_set_brightness": conf.state.last_set_brightness,  # note value from current state
    }

    try:
        LOGGER.debug("storing state...")
        statef = conf.state_f_path
        payload = json.dumps(
            data, indent=2, sort_keys=True, separators=(",", ": "), ensure_ascii=False
        )

        async with aiof.open(statef, mode="w") as f:
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


def unix_time_now() -> int:
    return int(datetime.now().timestamp())
