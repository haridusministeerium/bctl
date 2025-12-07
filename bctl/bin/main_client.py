#!/usr/bin/env python3

# TODO: consider pyro5 for rpc, as opposed to json over AF_UNIX

import click
import bctl.client as client
from bctl.common import Opts


def collect_per_disp_args(args: tuple[str, ...]) -> dict:
    p = {}
    # params = []
    for i, item in enumerate(args):
        if i % 2 == 0:
            assert item, "display ID must be given"
            d = item
        else:
            assert item.isdigit(), "display brightness must be a digit"
            v = int(item)
            # params.append([d, v])
            p[d] = v
    return p


def pack_opts(
    no_notify: bool = False,
    no_track: bool = False,
    no_internal: bool = False,
    no_external: bool = False,
    no_sync: bool = False,
) -> int:
    opts = 0
    if no_notify:
        opts |= Opts.NO_NOTIFY
    if no_track:
        opts |= Opts.NO_TRACK
    if no_internal:
        opts |= Opts.IGNORE_INTERNAL
    if no_external:
        opts |= Opts.IGNORE_EXTERNAL
    if no_sync:
        opts |= Opts.NO_SYNC
    return opts


@click.group
@click.pass_context
@click.option("--debug", is_flag=True, help="log at debug level")
def main(ctx, debug: bool):
    """Client for sending messages to BCTLD"""
    ctx.obj = client.Client(debug=debug)


@main.command
@click.pass_obj
@click.argument("delta", type=int, required=False)
def up(ctx, delta):
    """Bump up screens' brightness.

    :param ctx: context
    :param delta: % delta to bump brightness up by
    """
    assert delta is None or delta > 0, (
        "brightness % to bump up by needs to be positive int"
    )
    ctx.send_cmd(["up", delta])


@main.command
@click.pass_obj
@click.argument("delta", type=int, required=False)
def down(ctx, delta):
    """Bump down screens' brightness.

    :param ctx: context
    :param delta: % delta to bump brightness down by
    """
    assert delta is None or delta > 0, (
        "brightness % to bump down by needs to be positive int"
    )
    ctx.send_cmd(["down", delta])


@main.command
@click.pass_obj
@click.option("--no-notify", is_flag=True, help="do not send desktop notification")
@click.option("--no-track", is_flag=True, help="do not store set brightness in state")
@click.option("--no-external", is_flag=True, help="ingore external displays")
@click.option("--no-internal", is_flag=True, help="ingore internal displays")
@click.option("--no-sync", is_flag=True, help="skip brightness sync if enabled in config")
@click.argument("delta", type=int)
def delta(
    ctx,
    no_notify: bool,
    no_track: bool,
    no_external: bool,
    no_internal: bool,
    no_sync: bool,
    delta: int,
):
    """Change screens' brightness by given %

    :param ctx: context
    :param no_notify: do not send desktop notification on brightness change
    :param no_track: do not track this change in 'last_set_brightness'
    :param no_external: ignore external displays
    :param no_internal: ignore internal displays
    :param no_sync: skip brightness sync if enabled in config; useful with other
                    options that filter out some screens
    :param delta: % delta to change brightness down by
    """
    opts = pack_opts(
        no_notify=no_notify,
        no_track=no_track,
        no_internal=no_internal,
        no_external=no_external,
        no_sync=no_sync,
    )
    ctx.send_cmd(["delta", opts, delta])


@main.command
@click.pass_obj
@click.option("--no-notify", is_flag=True, help="do not send desktop notification")
@click.option("--no-track", is_flag=True, help="do not store set brightness in state")
@click.option("--no-external", is_flag=True, help="ingore external displays")
@click.option("--no-internal", is_flag=True, help="ingore internal displays")
@click.option("--no-sync", is_flag=True, help="skip brightness sync if enabled in config")
@click.argument("args", nargs=-1, type=str)
def set(
    ctx,
    no_notify: bool,
    no_track: bool,
    no_external: bool,
    no_internal: bool,
    no_sync: bool,
    args: tuple[str, ...],
):
    """Change screens' brightness to/by given %

    :param ctx: context
    :param no_notify: do not send desktop notification on brightness change
    :param no_track: do not track this change in 'last_set_brightness'
    :param no_external: ignore external displays
    :param no_internal: ignore internal displays
    :param no_sync: skip brightness sync if enabled in config; useful with other
                    options that filter out some screens
    :param value: % value to change brightness to/by
    """
    if not args:
        raise ValueError("params missing")
    elif len(args) == 1:
        value = args[0]
        opts = pack_opts(
            no_notify=no_notify,
            no_track=no_track,
            no_internal=no_internal,
            no_external=no_external,
            no_sync=no_sync,
        )

        if value.isdigit():
            ctx.send_cmd(["set", opts, int(value)])
        elif value.startswith(("-", "+")) and value[1:].isdigit():
            ctx.send_cmd(["delta", opts, int(value)])
        else:
            raise ValueError("brightness value to set needs to be [-+]?[0-9]+")
    else:
        assert len(args) % 2 == 0, (
            "when setting multiple displays, then even # or args expected"
        )
        # assert not (no_notify or no_track or no_external or no_internal or no_sync), (
            # "when setting brightnesses per monitor, then additional options are non-op"
        # )
        ctx.send_cmd(["set_for_async", collect_per_disp_args(args)])


@main.command
@click.pass_obj
@click.option("-r", "--retry", default=0, help="how many times to retry operation")
@click.option(
    "-s", "--sleep", default=0.5, help="how many seconds to sleep between retries"
)
@click.argument("args", nargs=-1, type=str)
def setvcp(ctx, retry: int, sleep: float | int, args: tuple[str, ...]):
    """Set VCP feature value(s) for all detected DDC displays

    :param ctx: context
    """
    assert len(args) >= 2, (
        "minimum 2 args needed, read ddcutil manual on [setvcp] command"
    )
    ctx.send_receive_cmd(["setvcp", retry, sleep, args])


@main.command("set-for")
@click.pass_obj
@click.option("-r", "--retry", default=0, help="how many times to retry operation")
@click.option(
    "-s", "--sleep", default=0.5, help="how many seconds to sleep between retries"
)
@click.argument("args", nargs=-1, type=str)
def set_for(ctx, retry: int, sleep: float | int, args: tuple[str, ...]):
    """similar to multi version of set(), but synchronized

    :param ctx: context
    :param value: % value to change brightness to/by
    """
    assert len(args) >= 2, (
        "minimum 2 args needed, read ddcutil manual on [set_for] command"
    )
    assert len(args) % 2 == 0, "even args required"
    ctx.send_receive_cmd(["set_for", retry, sleep, collect_per_disp_args(args)])


@main.command
@click.pass_obj
@click.option("-r", "--retry", default=0, help="how many times to retry operation")
@click.option(
    "-s", "--sleep", default=0.5, help="how many seconds to sleep between retries"
)
@click.argument("args", nargs=-1, type=str)
def getvcp(ctx, retry: int, sleep: float | int, args: tuple[str, ...]):
    """Get VCP feature value(s) for all detected DDC displays

    :param ctx: context
    """
    assert args, "minimum 1 feature needed, read ddcutil manual on [getvcp] command"
    ctx.send_receive_cmd(["getvcp", retry, sleep, args])


@main.command
@click.pass_obj
@click.option(
    "-i", "--individual", is_flag=True, help="retrieve brightness levels per screen"
)
@click.option("-r", "--raw", is_flag=True, help="retrieve raw brightness value")
@click.option("--no-external", is_flag=True, help="ingore external displays")
@click.option("--no-internal", is_flag=True, help="ingore internal displays")
@click.option("--no-offset", is_flag=True, help="do NOT normalize brightness value for effective offset")
def get(
    ctx,
    individual: bool,
    raw: bool,
    no_external: bool,
    no_internal: bool,
    no_offset: bool,
):
    """Get screens' brightness (%)

    :param ctx: context
    """
    if raw and not individual:
        raise ValueError(
            "raw values only make sense per-display, i.e. --raw option requires --individual"
        )
    opts = pack_opts(no_internal=no_internal, no_external=no_external)

    if individual:
        opts |= Opts.GET_INDIVIDUAL
    if raw:
        opts |= Opts.GET_RAW
    if no_offset:
        opts |= Opts.GET_NO_OFFSET_NORMALIZED
    ctx.send_receive_cmd(["get", opts])


@main.command
@click.pass_obj
@click.option("-r", "--retry", default=0, help="how many times to retry operation")
@click.option(
    "-s", "--sleep", default=0.5, help="how many seconds to sleep between retries"
)
def init(ctx, retry: int, sleep: float | int):
    """Re-initialize displays.

    :param ctx: context
    """
    ctx.send_cmd(["init", retry, sleep])


@main.command
@click.pass_obj
def sync(ctx):
    """Synchronize screens' brightness levels, even if disabled in the config.

    :param ctx: context
    """
    ctx.send_cmd(["sync"])


@main.command
@click.pass_obj
def kill(ctx):
    """Terminate the daemon process.

    :param ctx: context
    """
    ctx.send_cmd(["kill"])


if __name__ == "__main__":
    main()
