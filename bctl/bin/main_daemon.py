#!/usr/bin/env python3

import click
import bctl.daemon as daemon


@click.command
@click.option("--debug", is_flag=True, help="log at debug level")
@click.option("--state/--no-state", default=True, help="whether persisted state should be loaded")
def main(debug: bool, state: bool):
    daemon.main(debug, state)


if __name__ == "__main__":
    main()
