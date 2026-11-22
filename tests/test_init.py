import pytest
from conftest import FakeDisplay

from bctl import init
from bctl.config import Conf, MainDisplayCtl, OffsetConf, State
from bctl.display import DDCDisplay, DisplayType
from bctl.exceptions import CmdErr, FatalErr

DDCUTIL_DETECT = """\
Invalid display
   I2C bus:  /dev/i2c-4
   DRM connector:           card1-eDP-1
   EDID synopsis:
      Mfg id:               BOE - BOE

Display 1
   I2C bus:  /dev/i2c-11
   DRM connector:           card1-DP-1
   Monitor:                 AUS:PA278QV:L9GMQA215221

Display 2
   I2C bus:  /dev/i2c-12
   DRM connector:           card1-DP-2
   Monitor:                 DEL:U2720Q:ABC123

"""


def fake_run_cmd(monkeypatch, out: str = "", err: str = "", code: int = 0) -> list[list[str]]:
    calls: list[list[str]] = []

    async def _run_cmd(cmd, logger, throw_on_err=False):
        calls.append(cmd)
        return out, err, code

    monkeypatch.setattr(init, "run_cmd", _run_cmd)
    return calls


class TestDdcutilDetect:
    @pytest.fixture
    def internal(self, monkeypatch, conf):
        laptop = FakeDisplay("amdgpu_bl0", conf, dt=DisplayType.INTERNAL)

        async def resolve(_conf):
            return laptop

        monkeypatch.setattr(init, "resolve_single_internal_display_raw", resolve)
        return laptop

    async def test_parses_external_and_resolves_internal(self, monkeypatch, conf, internal):
        fake_run_cmd(monkeypatch, out=DDCUTIL_DETECT)
        displays = await init.get_ddcutil_displays(conf)

        assert displays[0] is internal
        ext = displays[1:]
        assert all(isinstance(d, DDCDisplay) for d in ext)
        assert [(d.id, d.bus) for d in ext] == [
            ("AUS:PA278QV:L9GMQA215221", "11"),
            ("DEL:U2720Q:ABC123", "12"),
        ]

    async def test_internal_skipped_when_ignored(self, monkeypatch, internal):
        fake_run_cmd(monkeypatch, out=DDCUTIL_DETECT)
        conf = Conf(state=State(), ignore_internal_display=True)
        displays = await init.get_ddcutil_displays(conf)
        assert internal not in displays
        assert len(displays) == 2

    async def test_unterminated_block_is_fatal(self, monkeypatch, conf):
        fake_run_cmd(monkeypatch, out="Display 1\n   I2C bus:  /dev/i2c-11\n   Monitor: X")
        with pytest.raises(FatalErr, match="not finalized"):
            await init.get_ddcutil_displays(conf)

    async def test_missing_i2c_module_is_fatal(self, monkeypatch, conf):
        fake_run_cmd(monkeypatch, err="ddcutil requires module i2c-dev", code=1)
        with pytest.raises(FatalErr, match="i2c-dev"):
            await init.get_ddcutil_displays(conf)

    async def test_other_failure_is_retriable_cmderr(self, monkeypatch, conf):
        fake_run_cmd(monkeypatch, err="something else", code=1)
        with pytest.raises(CmdErr):
            await init.get_ddcutil_displays(conf)


@pytest.fixture
def backlight_dir(tmp_path):
    d = tmp_path / "backlight"
    d.mkdir()
    (d / "amdgpu_bl0").mkdir()
    (d / "ddcci11").mkdir()
    (d / "dead").symlink_to(tmp_path / "nonexistent")
    return d


async def test_raw_displays_from_backlight_dir_skip_dead_links(backlight_dir):
    conf = Conf(state=State(), raw_device_dir=str(backlight_dir))
    displays = sorted(await init.get_raw_displays(conf), key=lambda d: d.id)
    assert [(d.id, d.type) for d in displays] == [
        ("amdgpu_bl0", DisplayType.INTERNAL),
        ("ddcci11", DisplayType.EXTERNAL),
    ]


async def test_brightnessctl_displays(monkeypatch, backlight_dir):
    conf = Conf(state=State(), raw_device_dir=str(backlight_dir))
    calls = fake_run_cmd(
        monkeypatch,
        out="amdgpu_bl0,backlight,120,47%,255\ngone,backlight,1,1%,100\n",
    )
    displays = await init.get_bctl_displays(conf)
    assert [d.id for d in displays] == ["amdgpu_bl0"]
    assert calls[0][0] == "brightnessctl"


class TestResolveDisplays:
    @pytest.fixture
    def detected(self, monkeypatch):
        """Make the ddcutil backend 'detect' whatever list the test assigns."""
        found: list[FakeDisplay] = []

        async def get(_conf):
            return list(found)

        monkeypatch.setattr(init, "get_ddcutil_displays", get)
        return found

    def two_displays(self, conf) -> list[FakeDisplay]:
        return [
            FakeDisplay("eDP", conf, dt=DisplayType.INTERNAL),
            FakeDisplay("ext", conf, dt=DisplayType.EXTERNAL),
        ]

    async def test_initializes_all_displays(self, conf, detected):
        detected.extend(self.two_displays(conf))
        displays = await init.resolve_displays(conf)
        assert all(d.is_percentage for d in displays)
        assert conf.main_display_ctl is MainDisplayCtl.DDCUTIL

    @pytest.mark.parametrize(
        "conf_kwargs, expected_ids",
        [
            ({"ignore_internal_display": True}, ["ext"]),
            ({"ignore_external_display": True}, ["eDP"]),
            ({"ignored_displays": {"laptop"}}, ["ext"]),
            ({"ignored_displays": {"ext"}}, ["eDP"]),
        ],
    )
    async def test_ignore_filters(self, detected, conf_kwargs, expected_ids):
        conf = Conf(state=State(), **conf_kwargs)
        detected.extend(self.two_displays(conf))
        assert [d.id for d in await init.resolve_displays(conf)] == expected_ids

    async def test_multiple_internal_displays_rejected(self, conf, detected):
        detected.extend(FakeDisplay(f"i{n}", conf, dt=DisplayType.INTERNAL) for n in range(2))
        with pytest.raises(RuntimeError, match="more than 1"):
            await init.resolve_displays(conf)

    @pytest.mark.parametrize(
        "offset_kwargs, ndisplays, offsets_kept",
        [
            ({}, 2, True),
            ({}, 1, False),
            ({"enabled_if_single_display": True}, 1, True),
            ({"enabled_if": "False"}, 2, False),
            ({"enabled_if": "len(displays) == 2"}, 2, True),
            ({"disabled_if": "len(displays) == 2"}, 2, False),
        ],
    )
    async def test_offset_enable_rules(self, detected, offset_kwargs, ndisplays, offsets_kept):
        conf = Conf(state=State(), offset=OffsetConf(offsets={"any": 10}, **offset_kwargs))
        detected.extend(self.two_displays(conf)[:ndisplays])
        displays = await init.resolve_displays(conf)
        assert all(d.offset == (10 if offsets_kept else 0) for d in displays)
