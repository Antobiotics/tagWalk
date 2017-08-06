import click

import fachung.logger as l

from fachung.commands.cli import BuilderCommand

@click.group()
def main(**kwargs):
    l.INFO("Starting TagWalk")

@main.command(cls=BuilderCommand)
def builder():
    l.INFO("Builder Command Detected")

if __name__ == '__main__':
    main()
