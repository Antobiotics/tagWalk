import click

import fachung.logger as l

import fachung.commands.cli as cli


@click.group()
def main(**kwargs):
    l.INFO("Starting TagWalk")

@main.command(cls=cli.BuilderCommand)
def builders():
    l.INFO("Builder Command Detected")

@main.command(cls=cli.ModelingCommand)
def modeling():
    l.INFO("Model Command Detected")

@main.command(cls=cli.ManagerCommand)
def managers():
    l.INFO("Managment Command Detected")

if __name__ == '__main__':
    main()
