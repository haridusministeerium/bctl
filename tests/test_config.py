import json

import pytest

from bctl.config import (
    STATE_VER,
    Conf,
    State,
    find_dupes,
    intersect,
    load_config,
    unix_time_now,
    write_state,
)
from bctl.exceptions import FatalErr


@pytest.fixture
def conf_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    d = tmp_path / "bctl"
    d.mkdir()
    return d


def write_json(path, data) -> None:
    path.write_text(json.dumps(data))


def test_defaults_without_config_files(conf_dir):
    conf = load_config()
    assert conf.brightness_step == 5
    assert conf.notify.enabled is True


def test_config_files_deep_merged_in_lexical_order(conf_dir):
    write_json(conf_dir / "20-b.json", {"brightness_step": 15, "notify": {"enabled": False}})
    write_json(conf_dir / "10-a.json", {"brightness_step": 10, "notify": {"timeout_ms": 1000}})
    conf = load_config()
    assert conf.brightness_step == 15
    assert conf.notify.timeout_ms == 1000
    assert conf.notify.enabled is False


def test_non_json_and_invalid_files_are_ignored(conf_dir):
    (conf_dir / "a.json").write_text("{not json")
    (conf_dir / "b.txt").write_text('{"brightness_step": 50}')
    assert load_config().brightness_step == 5


def test_duplicate_aliases_rejected(conf_dir):
    write_json(conf_dir / "c.json", {"aliases": {"d1": ["x"], "d2": ["x"]}})
    with pytest.raises(FatalErr, match="duplicates"):
        load_config()


@pytest.mark.parametrize("reserved", ["laptop", "internal"])
def test_reserved_aliases_rejected(conf_dir, reserved):
    write_json(conf_dir / "c.json", {"aliases": {"d1": [reserved]}})
    with pytest.raises(FatalErr, match="reserved"):
        load_config()


def test_offset_criteria_sorted_longest_first(conf_dir):
    write_json(
        conf_dir / "c.json",
        {"offset": {"offsets": {"any": 1, "id:longname": 2, "internal": 3}}},
    )
    assert list(load_config().offset.offsets) == ["id:longname", "internal", "any"]


def _write_state(path, **overrides) -> None:
    state = {"timestamp": unix_time_now(), "ver": STATE_VER, "last_set_brightness": 42}
    state.update(overrides)
    write_json(path, state)


def test_fresh_state_is_hydrated(conf_dir, tmp_path):
    state_f = tmp_path / "state.json"
    write_json(conf_dir / "c.json", {"state_f_path": str(state_f)})
    _write_state(state_f)
    assert load_config(load_state=True).state.last_set_brightness == 42
    assert load_config(load_state=False).state.last_set_brightness == -1


@pytest.mark.parametrize(
    "overrides",
    [{"timestamp": 1}, {"ver": STATE_VER + 1}],
    ids=["stale", "version-mismatch"],
)
def test_unusable_state_is_discarded(conf_dir, tmp_path, overrides):
    state_f = tmp_path / "state.json"
    write_json(conf_dir / "c.json", {"state_f_path": str(state_f)})
    _write_state(state_f, **overrides)
    assert load_config(load_state=True).state.last_set_brightness == -1


def test_state_age_check_disabled_with_zero(conf_dir, tmp_path):
    state_f = tmp_path / "state.json"
    write_json(conf_dir / "c.json", {"state_f_path": str(state_f), "max_state_age_sec": 0})
    _write_state(state_f, timestamp=1)
    assert load_config(load_state=True).state.last_set_brightness == 42


def test_missing_state_file_yields_default_state(conf_dir, tmp_path):
    write_json(conf_dir / "c.json", {"state_f_path": str(tmp_path / "nope.json")})
    assert load_config(load_state=True).state == State()


async def test_write_state_roundtrip(conf_dir, tmp_path):
    state_f = tmp_path / "state.json"
    write_json(conf_dir / "c.json", {"state_f_path": str(state_f)})
    conf = load_config()
    conf.state.last_set_brightness = 77
    conf.state.offsets = {"d1": [20, 15]}
    await write_state(conf)

    restored = load_config(load_state=True).state
    assert restored.last_set_brightness == 77
    assert restored.offsets == {"d1": [20, 15]}
    assert restored.timestamp > 0


async def test_write_state_failure_is_swallowed(tmp_path):
    conf = Conf(state=State(), state_f_path=str(tmp_path / "missing-dir" / "state.json"))
    await write_state(conf)  # must not raise


@pytest.mark.parametrize(
    "values, expected",
    [([1, 2, 3], []), ([3, 1, 2, 1], [1]), (["a", "b", "a", "b"], ["a", "b"])],
)
def test_find_dupes(values, expected):
    assert sorted(find_dupes(values)) == expected


def test_intersect():
    assert intersect([[1, 2, 3], [2, 3], (3, 2, 9)]) == {2, 3}
