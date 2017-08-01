from __future__ import absolute_import

import click

import tag_walk.logger as l

from tag_walk.asos import Asos
from tag_walk.commands.cli import pass_context


@click.command('asos', short_help="Prepares Asos Data")
@click.option('--df/--not-df', default=False)
@click.option('--labels/--not-labels', default=False)
@click.option('--images/--not-images', default=False)
@pass_context
def cli(ctx, df, labels, images):

    l.INFO("Preparing ASOS data")
    prep = Asos()
    prep.prepare(df=df,
                 labels=labels,
                 images=images)

