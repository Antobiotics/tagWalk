import click

import tag_walk.logger as l

@click.group()
def main(**kwargs):
    l.INFO("Starting TagWalk")


@main.command()
def run():
    l.INFO("Running")

if __name__ == '__main__':
    main()
