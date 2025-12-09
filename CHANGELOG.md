## 0.0.6 (unreleased)
---------------------

- getvcp/setvcp: add -d option to specify display(s)
  -d option to set/get vcp features for specific display(s) as opposed
   to all displays, as continues being the default


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
