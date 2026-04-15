## 0.0.12 (unreleased)
----------------------

- Nothing changed yet.


## 0.0.11 (2026-04-15)
----------------------

- fix common.wait_and_reraise() -- exception is now thrown on timeout;
  also unfinished tasks' cancellation is awaited on
- readme: mention user needs to be in `video` group, and display device
  brightness file must be writable by said group


## 0.0.10 (2026-04-11)
----------------------

- raise exception if `max_brightness_f` is not readable
- add note in readme to ensure internal display's device file's group
  is changed to `video`, and the user running the daemon is included in same group


## 0.0.9 (2026-03-26)
---------------------

- log caught exception stacktrace at debug level
- fix monitor model parsing from ddcutil's "Monitor:" line

## 0.0.8 (unreleased)
---------------------

release yanked


## 0.0.7 (2026-01-03)
---------------------

- client:
  - rename set-sync command to set-get
  - set-get gains the --raw option
  - make the two paths of set-get() return the same result, as in displays
    that were operated on will get queried for get_brightness()
  - client: all --retry/-r options lose the shorthand opt, as it collides w/ --raw/-r


## 0.0.6 (2025-12-09)
---------------------

- getvcp/setvcp: add -d option to specify display(s)
  -d option to set/get vcp features for specific display(s) as opposed
   to all displays, as continues being the default

- split set_for into set_for_sync & set_for_sync_all
  - note these are the commands triggered by set-sync (old name set-for) client command
    - set_for_sync_all makes set-sync client command to accept just a single
      int, which will then set all displays' brightnesses to that value;
      tracking, sync & notifications work as with "set"
    - set_for_sync acts as previous set-for command, i.e. accepts [diplay value] pairs


## 0.0.5 (2025-12-08)
---------------------

- config: survive pydantic model validation on State initialisation
- get: replace -i/--individual opts with -a/--all
- get: support querying brightness for specific display(s)


## 0.0.4 (2025-12-07)
---------------------

- automatically assign [display,laptop] aliases to
  DisplayType.INTERNAL -- i.e. laptop -- screen
- improve documentation in config.py


## 0.0.3 (2025-12-06)
---------------------

- set & delta commands: add --track/--no-track options
- get command: do not use last_set_brightness state
  and return the _actual_ current brightness of
  displays
- start using pydantic for config models
- introduce offset feature
  allows offsetting certain dispalys' brightnesses from others.
  useful in cases where some displays, e.g. that of laptop's internal
  one, is brighter or dimmer than others
- allow multiple configuration files
  config is now merged from multiple files from `$XDG_CONFIG_HOME/bctl/*.json`



## 0.0.2 (2025-11-21)
---------------------

- adds retry mechanism - depends on retry-deco pkg
- 'set' command:
  - now supports delta args - need to be prefixed w/ +/-
  - support per-monitor values, e.g. $ bctl set 'Display 1' 30 'Display 2' 100
  - now as --no-notify option to disablechange notif
- 'delta' command:
  - now as --no-notify option to disablechange notif
- removed 'init-block' command
- added 'set-for' command
  - similar to new per-/multimon syntax of 'set', but synchronzied & retriable
- minimum required py ver now 3.12 (due to generics' bracket syntax)
- add mise.toml
- sync_strategy config item now list of stragegies
  - first matched/functional startegy gets applied
- add github actions ci/cd
