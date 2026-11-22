import os
import shutil
import tempfile

# bctl.common resolves runtime/cache paths at import time, so redirect the XDG
# dirs before any bctl module gets imported; keeps tests off the real socket & state.
_TMP_ROOT = tempfile.mkdtemp(prefix="bctl-tests-")
for _var in ("XDG_RUNTIME_DIR", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
    _path = os.path.join(_TMP_ROOT, _var.lower())
    os.makedirs(_path)
    os.environ[_var] = _path

import pytest

from bctl.config import Conf, State
from bctl.display import BackendType, Display, DisplayType


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)


class FakeDisplay(Display):
    """In-memory display recording every raw value written to the 'hardware'."""

    def __init__(
        self,
        id: str,
        conf: Conf,
        dt: DisplayType = DisplayType.EXTERNAL,
        bt: BackendType = BackendType.SIM,
        raw: int = 50,
        max_brightness: int = 100,
    ) -> None:
        super().__init__(id, dt, bt, conf)
        self.raw_brightness = raw
        self.max_brightness = max_brightness
        self.writes: list[int] = []
        self.fail_with: Exception | None = None

    async def init(self) -> None:
        self._init()

    async def _set_brightness(self, value: int) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        self.writes.append(value)


class FakeNotif:
    def __init__(self) -> None:
        self.changes: list[int] = []
        self.errors: list[Exception] = []

    async def notify_change(self, val: int) -> None:
        self.changes.append(val)

    async def notify_err(self, err: Exception) -> None:
        self.errors.append(err)


@pytest.fixture
def conf(tmp_path) -> Conf:
    return Conf(state=State(), state_f_path=str(tmp_path / "bctld.state"))


@pytest.fixture
def make_display(conf):
    """Factory for initialized FakeDisplays; pass conf= to override the shared one."""

    def _make(id: str = "d1", **kwargs) -> FakeDisplay:
        c = kwargs.pop("conf", conf)
        d = FakeDisplay(id, c, **kwargs)
        d._init()
        return d

    return _make
