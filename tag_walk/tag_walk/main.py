import click

import tag_walk.logger as l

import tag_walk.data as data

@click.group()
def main(**kwargs):
    l.INFO("Starting TagWalk")


@main.command()
@click.option('--df/--not-df', default=False)
@click.option('--labels/--not-labels', default=False)
@click.option('--images/--not-images', default=False)
def paperdoll_prepare(df, labels, images):

    l.INFO("Preparing Paperdoll data")
    paperdoll = data.PaperDoll(readable_labels=False)
    paperdoll.prepare(df=df,
                      labels=labels,
                      images=images)

if __name__ == '__main__':
    main()
