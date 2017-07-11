import click

import tag_walk.logger as l

from tag_walk.commands.cli import BuilderCommand

@click.group()
def main(**kwargs):
    l.INFO("Starting TagWalk")

@main.command(cls=BuilderCommand)
def builder():
    l.INFO("Builder Command Detected")

if __name__ == '__main__':
    main()
