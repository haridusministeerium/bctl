import json
import socket
import threading
from typing import ClassVar

import pytest
from click.testing import CliRunner

from bctl import client as client_mod
from bctl.bin import main_client
from bctl.client import Client
from bctl.common import Opts


class RecordingClient:
    sent: ClassVar[list[tuple[str, list]]] = []

    def __init__(self, debug=False):
        pass

    def send_cmd(self, cmd):
        self.sent.append(("send", cmd))

    def send_receive_cmd(self, cmd):
        self.sent.append(("send_receive", cmd))


@pytest.fixture
def cli(monkeypatch):
    RecordingClient.sent = []
    monkeypatch.setattr(main_client.client, "Client", RecordingClient)
    runner = CliRunner()

    def invoke(*args):
        result = runner.invoke(main_client.main, list(args))
        return result, RecordingClient.sent

    return invoke


@pytest.mark.parametrize(
    "args, expected",
    [
        (["up"], ("send", ["up", None])),
        (["up", "20"], ("send", ["up", 20])),
        (["down"], ("send", ["down", None])),
        (["delta", "--", "-20"], ("send", ["delta", 0, -20])),
        (["delta", "--no-notify", "5"], ("send", ["delta", Opts.NO_NOTIFY, 5])),
        (["set", "55"], ("send", ["set", 0, 55])),
        (["set", "+20"], ("send", ["delta", 0, 20])),
        (["set", "--", "-20"], ("send", ["delta", 0, -20])),
        (["set", "--no-internal", "--no-sync", "40"], ("send", ["set", Opts.IGNORE_INTERNAL | Opts.NO_SYNC, 40])),
        (["set", "laptop", "30", "ext", "70"], ("send", ["set_for_async", {"laptop": 30, "ext": 70}])),
        (["get"], ("send_receive", ["get", 0, ()])),
        (["get", "laptop"], ("send_receive", ["get", 0, ("laptop",)])),
        (["get", "--all", "--raw"], ("send_receive", ["get", Opts.GET_ALL | Opts.GET_RAW, ()])),
        (["set-get", "40"], ("send_receive", ["set_get", 0, 0.5, 0, 40])),
        (["set-get", "laptop", "30"], ("send_receive", ["set_get_for", 0, 0.5, 0, {"laptop": 30}])),
        (["init"], ("send", ["init", 0, 0.5])),
        (["init", "--retry", "2", "-s", "1"], ("send", ["init", 2, 1.0])),
        (["sync"], ("send", ["sync"])),
        (["kill"], ("send", ["kill", 0])),
        (["kill", "3"], ("send", ["kill", 3])),
        (["setvcp", "D6", "01"], ("send_receive", ["setvcp", 0, 0.5, (), ("D6", "01")])),
        (["getvcp", "-d", "ext", "10"], ("send_receive", ["getvcp", 0, 0.5, ("ext",), ("10",)])),
    ],
)
def test_cli_builds_protocol_messages(cli, args, expected):
    result, sent = cli(*args)
    assert result.exit_code == 0, result.output
    assert sent == [expected]


@pytest.mark.parametrize(
    "args",
    [
        ["up", "0"],
        ["set", "abc"],
        ["set", "laptop"],
        ["set", "laptop", "101", "x", "1"],
        ["get", "--raw"],
        ["get", "--all", "laptop"],
        ["setvcp", "D6"],
        ["getvcp"],
    ],
)
def test_cli_rejects_invalid_args(cli, args):
    result, sent = cli(*args)
    assert result.exit_code != 0
    assert sent == []


def test_set_get_rejects_out_of_range(cli):
    result, sent = cli("set-get", "120")
    assert result.exit_code != 0
    assert sent == []


@pytest.fixture
def fake_daemon(tmp_path, monkeypatch):
    """One-shot unix socket server in a thread, replying with a canned payload."""
    path = str(tmp_path / "s.sock")
    monkeypatch.setattr(client_mod, "SOCKET_PATH", path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)
    received: list = []

    def serve(reply: bytes):
        conn, _ = srv.accept()
        with conn:
            chunks = []
            while chunk := conn.recv(4096):
                chunks.append(chunk)
            received.append(json.loads(b"".join(chunks)))
            if reply:
                conn.sendall(reply)

    def start(reply: bytes) -> threading.Thread:
        t = threading.Thread(target=serve, args=(reply,), daemon=True)
        t.start()
        return t

    yield start, received
    srv.close()


@pytest.mark.parametrize(
    "reply, code, stream, printed",
    [
        (b'[0,55]', 0, "out", "55\n"),
        (b'[0,["eDP",50],["ext",60]]', 0, "out", '["eDP",50]\n["ext",60]\n'),
        (b'[1,"no displays"]', 1, "err", '"no displays"\n'),
    ],
)
def test_client_send_receive(fake_daemon, capsys, reply, code, stream, printed):
    start, received = fake_daemon
    t = start(reply)
    with pytest.raises(SystemExit) as exc:
        Client().send_receive_cmd(["get"])
    t.join(timeout=2)

    assert received == [["get"]]
    assert exc.value.code == code
    assert getattr(capsys.readouterr(), stream) == printed


def test_client_send_fire_and_forget(fake_daemon):
    start, received = fake_daemon
    t = start(b"")
    Client().send_cmd(["up", None])
    t.join(timeout=2)
    assert received == [["up", None]]


def test_client_exits_when_daemon_not_running(tmp_path, monkeypatch):
    monkeypatch.setattr(client_mod, "SOCKET_PATH", str(tmp_path / "missing.sock"))
    with pytest.raises(SystemExit) as exc:
        Client().send_cmd(["sync"])
    assert exc.value.code == 1


def test_pack_opts():
    assert main_client.pack_opts() == 0
    assert main_client.pack_opts(no_notify=True, no_external=True) == Opts.NO_NOTIFY | Opts.IGNORE_EXTERNAL


def test_collect_per_disp_args():
    assert main_client.collect_per_disp_args(("laptop", "0", "ext", "100")) == {"laptop": 0, "ext": 100}
    with pytest.raises(AssertionError):
        main_client.collect_per_disp_args(("ext", "high"))
