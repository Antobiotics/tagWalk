from __future__ import absolute_import

import click

import tag_walk.logger as l

from tag_walk.paperdoll import PaperDoll
from tag_walk.commands.cli import pass_context


@click.command('paperdoll', short_help="Prepares Paperdoll Data")
@click.option('--df/--not-df', default=False)
@click.option('--labels/--not-labels', default=False)
@click.option('--images/--not-images', default=False)
@pass_context
def cli(ctx, df, labels, images):

    l.INFO("Preparing ASOS data")
    prep = PaperDoll()
    prep.prepare(df=df,
                 labels=labels,
                 images=images)

