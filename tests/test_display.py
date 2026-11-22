from typing import ClassVar

import pytest
from conftest import FakeDisplay

from bctl.common import Opts
from bctl.config import Conf, OffsetConf, OffsetType, State
from bctl.display import (
    BCTLDisplay,
    BrilloDisplay,
    DDCDisplay,
    DisplayType,
    RawDisplay,
    get_disp_filter,
)
from bctl.exceptions import FatalErr


def offset_conf(offsets: dict[str, int], type=OffsetType.HARD, **kwargs) -> Conf:
    return Conf(state=State(), offset=OffsetConf(type=type, offsets=offsets), **kwargs)


class TestInitValidation:
    def test_empty_id_is_fatal(self, conf):
        with pytest.raises(FatalErr):
            FakeDisplay("", conf)

    @pytest.mark.parametrize(
        "raw, max_b",
        [(-1, 100), (10, 0), (101, 100)],
        ids=["raw-uninitialized", "max-uninitialized", "raw-above-max"],
    )
    def test_invalid_brightness_state_is_fatal(self, conf, raw, max_b):
        d = FakeDisplay("d1", conf, raw=raw, max_brightness=max_b)
        with pytest.raises(FatalErr):
            d._init()

    def test_internal_display_gets_builtin_aliases(self, make_display):
        assert make_display("eDP", dt=DisplayType.INTERNAL).names == {"eDP", "laptop", "internal"}

    def test_configured_aliases_added_to_names(self):
        conf = Conf(state=State(), aliases={"d1": ("left", "main")})
        assert FakeDisplay("d1", conf).names == {"d1", "left", "main"}


class TestBrightnessNoOffset:
    async def test_set_writes_and_returns_new_value(self, make_display):
        d = make_display(raw=50)
        assert await d.set_brightness(70) == 70
        assert d.writes == [70]
        assert d.get_brightness() == 70

    @pytest.mark.parametrize("value, expected", [(150, 100), (-5, 0)])
    async def test_set_clamps_to_0_100(self, make_display, value, expected):
        d = make_display(raw=50)
        assert await d.set_brightness(value) == expected
        assert d.writes == [expected]

    async def test_unchanged_value_skips_hardware_write(self, make_display):
        d = make_display(raw=50)
        assert await d.set_brightness(50) == 50
        assert d.writes == []

    async def test_adjust_is_relative(self, make_display):
        d = make_display(raw=50)
        assert await d.adjust_brightness(-15) == 35
        assert await d.adjust_brightness(100) == 100

    async def test_non_percentage_device_converts_raw_values(self, make_display):
        d = make_display(raw=128, max_brightness=255)
        assert d.is_percentage is False
        assert d.get_brightness() == 50
        assert d.get_brightness(raw=True) == 128

        assert await d.set_brightness(20) == 20
        assert d.writes == [51]
        assert d.get_brightness_for_opts(Opts.GET_RAW) == 51

        await d.set_brightness(100)
        assert d.writes[-1] == 255


class TestOffsets:
    @pytest.mark.parametrize(
        "crit, dt, id, expected",
        [
            ("any", DisplayType.EXTERNAL, "d1", 10),
            ("internal", DisplayType.INTERNAL, "eDP", 10),
            ("internal", DisplayType.EXTERNAL, "d1", 0),
            ("external", DisplayType.EXTERNAL, "d1", 10),
            ("external", DisplayType.INTERNAL, "eDP", 0),
            ("id:d1", DisplayType.EXTERNAL, "d1", 10),
            ("id:laptop", DisplayType.INTERNAL, "eDP", 10),
            ("id:other", DisplayType.EXTERNAL, "d1", 0),
        ],
    )
    def test_criteria_matching(self, crit, dt, id, expected):
        assert FakeDisplay(id, offset_conf({crit: 10}), dt=dt).offset == expected

    def test_first_matching_criteria_wins(self):
        d = FakeDisplay("d1", offset_conf({"id:d1": -5, "any": 10}))
        assert d.offset == -5

    def test_misconfigured_criteria_is_fatal(self):
        with pytest.raises(FatalErr, match="misconfigured offset"):
            FakeDisplay("d1", offset_conf({"bogus": 10}))

    async def test_hard_positive_offset(self, make_display):
        conf = offset_conf({"any": 20})
        d = make_display(conf=conf, raw=0)

        assert await d.set_brightness(50) == 50
        assert d.writes[-1] == 70
        assert d.get_brightness(no_offset_normalized=True) == 70
        assert conf.state.offsets["d1"] == [20, 20]

        # offset gets squeezed near the top...
        assert await d.set_brightness(90) == 90
        assert d.writes[-1] == 100
        assert d.eoffset == 10
        # ...and is a hard floor at the bottom
        assert await d.set_brightness(0) == 0
        assert d.writes[-1] == 20

    async def test_hard_negative_offset(self, make_display):
        d = make_display(conf=offset_conf({"any": -20}), raw=50)
        assert await d.set_brightness(50) == 50
        assert d.writes[-1] == 30
        assert await d.set_brightness(10) == 10
        assert d.writes[-1] == 0
        assert d.eoffset == -10

    async def test_soft_offset_can_go_below_offset_floor(self, make_display):
        d = make_display(conf=offset_conf({"any": 20}, type=OffsetType.SOFT), raw=0)
        await d.set_brightness(50)  # raw 70, eoffset 20
        await d.adjust_brightness(-60)  # asks for -10%
        assert d.writes[-1] == 10
        assert d.get_brightness() == 0

    async def test_hard_offset_keeps_floor_on_overshoot(self, make_display):
        d = make_display(conf=offset_conf({"any": 20}), raw=0)
        await d.set_brightness(50)
        await d.adjust_brightness(-60)
        assert d.writes[-1] == 20

    def test_eoffset_hydrated_from_state_when_offset_unchanged(self):
        conf = offset_conf({"any": 20})
        conf.state.offsets = {"d1": [20, 5]}
        assert FakeDisplay("d1", conf).eoffset == 5

    def test_state_discarded_when_offset_removed(self):
        conf = offset_conf({})
        conf.state.offsets = {"d1": [20, 5]}
        d = FakeDisplay("d1", conf)
        assert d.eoffset == 0
        assert "d1" not in conf.state.offsets

    def test_state_replaced_when_offset_changed(self):
        conf = offset_conf({"any": 30})
        conf.state.offsets = {"d1": [20, 5]}
        d = FakeDisplay("d1", conf)
        assert d.eoffset == 0
        assert conf.state.offsets["d1"] == [30, 0]


@pytest.mark.parametrize(
    "opts, internal_kept, external_kept",
    [
        (0, True, True),
        (Opts.IGNORE_INTERNAL, False, True),
        (Opts.IGNORE_EXTERNAL, True, False),
        (Opts.IGNORE_INTERNAL | Opts.IGNORE_EXTERNAL, False, False),
    ],
)
def test_get_disp_filter(make_display, opts, internal_kept, external_kept):
    f = get_disp_filter(opts)
    assert f(make_display("i", dt=DisplayType.INTERNAL)) is internal_kept
    assert f(make_display("e", dt=DisplayType.EXTERNAL)) is external_kept


@pytest.mark.parametrize(
    "id, expected", [("ddcci11", DisplayType.EXTERNAL), ("amdgpu_bl0", DisplayType.INTERNAL)]
)
def test_non_ddc_display_type_from_id(conf, id, expected):
    assert BrilloDisplay(id, conf).type is expected


class TestDDCDisplay:
    BLOCK: ClassVar[list[str]] = ["Display 1", "I2C bus:  /dev/i2c-11", "DRM connector:    card1-DP-1", "Monitor:   AUS:PA278QV:L9GMQA215221"]

    @pytest.fixture
    def cmds(self, monkeypatch):
        calls: list[list[str]] = []

        async def fake_run_cmd(cmd, logger, throw_on_err=False):
            calls.append(cmd)
            return "VCP 10 C 40 100\n", "", 0

        monkeypatch.setattr("bctl.display.run_cmd", fake_run_cmd)
        return calls

    def test_parses_detect_block(self, conf):
        d = DDCDisplay(self.BLOCK, conf)
        assert (d.id, d.bus, d.type) == ("AUS:PA278QV:L9GMQA215221", "11", DisplayType.EXTERNAL)

    async def test_init_reads_brightness_via_getvcp(self, conf, cmds):
        d = DDCDisplay(self.BLOCK, conf)
        await d.init()
        assert (d.raw_brightness, d.max_brightness) == (40, 100)
        assert cmds == [["ddcutil", "--brief", "getvcp", "10", "--bus", "11"]]

    async def test_set_brightness_invokes_setvcp(self, conf, cmds):
        d = DDCDisplay(self.BLOCK, conf)
        await d.init()
        await d.set_brightness(65)
        assert cmds[-1] == ["ddcutil", "--skip-ddc-checks", "setvcp", "10", "65", "--bus", "11"]


async def test_bctl_display_parses_machine_readable_line(conf):
    d = BCTLDisplay("amdgpu_bl0,backlight,120,47%,255", conf)
    await d.init()
    assert (d.id, d.type, d.raw_brightness, d.max_brightness) == ("amdgpu_bl0", DisplayType.INTERNAL, 120, 255)


class TestRawDisplay:
    @pytest.fixture
    def device(self, tmp_path):
        dev = tmp_path / "amdgpu_bl0"
        dev.mkdir()
        (dev / "brightness").write_text("120\n")
        (dev / "max_brightness").write_text("255\n")
        return dev

    async def test_init_and_set_via_sysfs_files(self, conf, device):
        d = RawDisplay(str(device), conf)
        await d.init()
        assert (d.id, d.type, d.raw_brightness, d.max_brightness) == ("amdgpu_bl0", DisplayType.INTERNAL, 120, 255)

        await d.set_brightness(100)
        assert (device / "brightness").read_text() == "255"

    async def test_missing_max_brightness_is_fatal(self, conf, device):
        (device / "max_brightness").unlink()
        with pytest.raises(FatalErr, match="max_brightness"):
            await RawDisplay(str(device), conf).init()

    async def test_unwritable_brightness_is_fatal(self, conf, device):
        (device / "brightness").chmod(0o444)
        with pytest.raises(FatalErr, match="RW perms"):
            await RawDisplay(str(device), conf).init()
