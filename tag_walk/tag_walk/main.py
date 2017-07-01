import click

import tag_walk.logger as l

import tag_walk.data as data

@click.group()
def main(**kwargs):
    l.INFO("Starting TagWalk")


@main.command()
def prepare():

    l.INFO("Preparing Paperdoll data")
    paperdoll = data.PaperDoll()
    paperdoll.save_df()
    paperdoll.save_labels()

if __name__ == '__main__':
    main()
