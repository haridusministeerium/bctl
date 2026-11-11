# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`bctl` is a Linux brightness-control daemon (`bctld`) plus a CLI client (`bctl`) that talk over a unix socket at `$XDG_RUNTIME_DIR/bctl/bctld-ipc.sock`. Python >=3.12 (uses PEP 695 generics, `except*`, `match`). User-facing docs and config semantics live in `README.md`; all config keys with defaults/descriptions are in `bctl/config.py` (`Conf`).

## Commands

The venv is managed by mise (`mise.toml` creates `.venv` via uv).

```sh
pip install -e '.[dev,test]' # installs ruff, ty, ipdb, pytest, pytest-asyncio
ruff check bctl tests        # lint (line-length 120)
ty check bctl                # type check
pytest                       # run all tests
pytest tests/test_display.py::TestOffsets::test_hard_positive_offset  # single test
bctld --debug                # run real daemon
bctlsim --debug -w 0.2 -n 3  # run daemon against simulated displays (no hardware needed)
bctlsim -w 0.2 -f s -e 1     # simulate failures: -f i (init) | s (set), -e exit code
bctl --help                  # client; e.g. `bctl get`, `bctl set 50`, `bctl init`
```

Tests need no hardware, external binaries or dbus.
- `tests/conftest.py` redirects the `XDG_*` dirs to a temp dir before importing `bctl`, because `bctl.common` resolves the socket and cache paths at import time.
- It also provides `FakeDisplay`, an in-memory `Display` that records raw writes, plus `FakeNotif`.
- Daemon tests monkeypatch the module globals in `bctl.daemon` (`CONF`, `DISPLAYS`, `TASK_QUEUE`, `LOCK`, `NOTIF`, `init_displays`).
- External commands are faked by monkeypatching `run_cmd` in the module under test.
- pytest-asyncio runs in auto mode, so `async def test_*` needs no marker.

For manual end-to-end checks, `bctlsim` (`bctl/bin/main_daemon_sim.py` → `SimulatedDisplay`) runs the daemon without real displays. Only one daemon instance can run at a time (`tendo.singleton`).

Releases use zest.releaser (`CHANGELOG.md` in md format, no PyPI upload from release step; CI in `.github/workflows/build-release.yml` delegates to a shared workflow). Add user-visible changes to the top "unreleased" section of `CHANGELOG.md`.

## Architecture

**Daemon (`bctl/daemon.py`)** uses module-level globals (`DISPLAYS`, `TASK_QUEUE`, `LOCK`, `CONF`, `NOTIF`) initialised in `main()`. `run()` starts an `asyncio.TaskGroup` with:
- `process_client_commands` – unix socket server. Each connection sends one JSON array (e.g. `["get"]`, `["set", opts, 50]`); most commands have several arities (e.g. `["kill"]` / `["kill", code]`, `["get"]` / `["get", opts, displays]`), so treat the `match` cases in `daemon.py` and `execute_tasks` as the authoritative list.
  - Queued, no reply: `up`, `down`, `delta`, `set`, `init`, `sync`, `kill` go onto `TASK_QUEUE`.
  - Inline under `LOCK`, reply with `[code, *payload]`: `get`, `set_get`, `set_get_for`, `setvcp`, `getvcp`.
  - `set_for_async` closes the socket immediately (no reply), then sets per-display brightness inline under `LOCK`.
- `process_q` – batches queue items arriving within `msg_consumption_window_sec` and hands them to `execute_tasks`, which coalesces them (deltas summed, a `set` cancels prior deltas) into a single display operation. This is why spamming a key runs ddcutil only once.
- udev `drm change` monitor (debounced) and optional `periodic_init`, both enqueue `["init", retry, sleep]`.
- signal handlers: SIGUSR1/SIGUSR2 bump brightness, SIGINT/SIGTERM terminate. State (`State`: last set brightness, offsets) is persisted to the cache dir in `finally` and rehydrated on start.

The client wire protocol is defined implicitly by the `match` cases in `daemon.py` and the `send_cmd`/`send_receive_cmd` calls in `bctl/bin/main_client.py` — change both together. `Opts` (`bctl/common.py`) is an `IntFlag` bitmask passed through the protocol.

**Displays (`bctl/display.py`)**: abstract `Display` holds cached `raw_brightness`, offsets (`offset`/`eoffset`, soft vs hard), and percent↔raw conversion; subclasses implement `init()` / `_set_brightness()` per backend: `DDCDisplay` (ddcutil, external), `RawDisplay` (`/sys/class/backlight`), `BCTLDisplay` (brightnessctl), `BrilloDisplay`, `SimulatedDisplay`. Brightness reads come from the cache, not the hardware. `get_disp_filter(opts)` builds display filters from `Opts`.

**Discovery (`bctl/init.py`)**: `resolve_displays(conf)` picks backends from `main_display_ctl` (plus `internal_display_ctl` for the laptop panel when using ddcutil), applies ignore filters, and disables offsets per `offset.enabled_if`/`disabled_if` — these are Python expressions run through `eval()`.

**Config (`bctl/config.py`)**: pydantic models; all `$XDG_CONFIG_HOME/bctl/*.json` files are deep-merged in lexical order. Bump `STATE_VER` whenever the `State` schema changes.

**Errors (`bctl/exceptions.py`)** drive daemon control flow:
- `RetriableErr` subclasses (`CmdErr`, `PayloadErr`, `ExitableErr`) are what `retry_deco.RetryAsync` retries on; failed display ops typically trigger `init_displays` on the last retry.
- `PayloadErr` carries the payload returned to the client.
- `ExitableErr` / `FatalErr` escaping the TaskGroup make the daemon exit (`FatalErr` → `fatal_exit_code`, default 100, meant for systemd `RestartPreventExitStatus`).
- `notify=False` suppresses the desktop notification (`bctl/notify.py`).

Use `common.run_cmd` for external commands and `common.wait_and_reraise` for awaiting task groups with timeout and cancellation.
