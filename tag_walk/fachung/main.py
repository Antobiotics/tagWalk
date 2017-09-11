import click

import fachung.logger as l

from fachung.commands.cli import BuilderCommand
from fachung.commands.cli import ModelingCommand

@click.group()
def main(**kwargs):
    l.INFO("Starting TagWalk")

@main.command(cls=BuilderCommand)
def builders():
    l.INFO("Builder Command Detected")

@main.command(cls=ModelingCommand)
def modeling():
    l.INFO("Model Command Detected")

if __name__ == '__main__':
    main()
