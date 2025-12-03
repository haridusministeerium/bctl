## 0.0.3 (unreleased)
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
